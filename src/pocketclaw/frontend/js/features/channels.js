/**
 * PocketPaw - Channels Feature Module
 *
 * Created: 2026-02-06
 *
 * Contains channel management state and methods:
 * - Channel status polling
 * - Save channel configuration (tokens)
 * - Start/Stop channel adapters dynamically
 * - WhatsApp personal mode QR polling
 */

window.PocketPaw = window.PocketPaw || {};

window.PocketPaw.Channels = {
    /**
     * Get initial state for Channels
     */
    getState() {
        return {
            showChannels: false,
            channelsTab: 'discord',
            channelStatus: {
                discord: { configured: false, running: false },
                slack: { configured: false, running: false },
                whatsapp: { configured: false, running: false, mode: 'personal' },
                telegram: { configured: false, running: false }
            },
            channelForms: {
                discord: { bot_token: '' },
                slack: { bot_token: '', app_token: '' },
                whatsapp: { access_token: '', phone_number_id: '', verify_token: '' },
                telegram: { bot_token: '' }
            },
            channelLoading: false,
            // WhatsApp personal mode QR state
            whatsappQr: null,
            whatsappConnected: false,
            whatsappQrPolling: null
        };
    },

    /**
     * Get methods for Channels
     */
    getMethods() {
        return {
            /**
             * Open Channels modal and fetch status
             */
            async openChannels() {
                this.showChannels = true;
                await this.getChannelStatus();
                this.startWhatsAppQrPollingIfNeeded();
                this.$nextTick(() => {
                    if (window.refreshIcons) window.refreshIcons();
                });
            },

            /**
             * Fetch channel status from backend
             */
            async getChannelStatus() {
                try {
                    const res = await fetch('/api/channels/status');
                    if (res.ok) {
                        this.channelStatus = await res.json();
                    }
                } catch (e) {
                    console.error('Failed to get channel status', e);
                }
            },

            /**
             * Save channel config (tokens) to backend
             */
            async saveChannelConfig(channel) {
                this.channelLoading = true;
                try {
                    const res = await fetch('/api/channels/save', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            channel,
                            config: this.channelForms[channel]
                        })
                    });
                    const data = await res.json();
                    if (data.status === 'ok') {
                        this.showToast(`${channel.charAt(0).toUpperCase() + channel.slice(1)} config saved!`, 'success');
                        // Clear form inputs after save
                        for (const key in this.channelForms[channel]) {
                            this.channelForms[channel][key] = '';
                        }
                        await this.getChannelStatus();
                    } else {
                        this.showToast(data.error || 'Failed to save', 'error');
                    }
                } catch (e) {
                    this.showToast('Failed to save config: ' + e.message, 'error');
                } finally {
                    this.channelLoading = false;
                }
            },

            /**
             * Save WhatsApp mode (personal/business)
             */
            async saveWhatsAppMode(mode) {
                this.channelLoading = true;
                try {
                    // Stop adapter if running (mode change requires restart)
                    if (this.channelStatus.whatsapp?.running) {
                        await fetch('/api/channels/toggle', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ channel: 'whatsapp', action: 'stop' })
                        });
                    }

                    const res = await fetch('/api/channels/save', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            channel: 'whatsapp',
                            config: { mode }
                        })
                    });
                    const data = await res.json();
                    if (data.status === 'ok') {
                        this.showToast(`WhatsApp mode set to ${mode}`, 'success');
                        await this.getChannelStatus();
                        this.whatsappQr = null;
                        this.whatsappConnected = false;
                        this.startWhatsAppQrPollingIfNeeded();
                    }
                } catch (e) {
                    this.showToast('Failed to save mode: ' + e.message, 'error');
                } finally {
                    this.channelLoading = false;
                    this.$nextTick(() => {
                        if (window.refreshIcons) window.refreshIcons();
                    });
                }
            },

            /**
             * Toggle (start/stop) a channel adapter
             */
            async toggleChannel(channel) {
                this.channelLoading = true;
                const isRunning = this.channelStatus[channel]?.running;
                const action = isRunning ? 'stop' : 'start';

                try {
                    const res = await fetch('/api/channels/toggle', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ channel, action })
                    });
                    const data = await res.json();

                    if (data.error) {
                        this.showToast(data.error, 'error');
                    } else {
                        const label = channel.charAt(0).toUpperCase() + channel.slice(1);
                        this.showToast(
                            action === 'start' ? `${label} started!` : `${label} stopped.`,
                            action === 'start' ? 'success' : 'info'
                        );
                        await this.getChannelStatus();

                        // Start/stop QR polling for WhatsApp personal mode
                        if (channel === 'whatsapp') {
                            if (action === 'start') {
                                this.startWhatsAppQrPollingIfNeeded();
                            } else {
                                this.stopWhatsAppQrPolling();
                                this.whatsappQr = null;
                                this.whatsappConnected = false;
                            }
                        }
                    }
                } catch (e) {
                    this.showToast('Failed to toggle channel: ' + e.message, 'error');
                } finally {
                    this.channelLoading = false;
                    this.$nextTick(() => {
                        if (window.refreshIcons) window.refreshIcons();
                    });
                }
            },

            /**
             * Start QR polling if WhatsApp is running in personal mode
             */
            startWhatsAppQrPollingIfNeeded() {
                this.stopWhatsAppQrPolling();
                const isPersonal = this.channelStatus.whatsapp?.mode === 'personal';
                const isRunning = this.channelStatus.whatsapp?.running;
                if (isPersonal && isRunning && !this.whatsappConnected) {
                    this.pollWhatsAppQr();
                    this.whatsappQrPolling = setInterval(() => this.pollWhatsAppQr(), 2000);
                }
            },

            /**
             * Poll the WhatsApp QR endpoint
             */
            async pollWhatsAppQr() {
                try {
                    const res = await fetch('/api/whatsapp/qr');
                    if (res.ok) {
                        const data = await res.json();
                        this.whatsappQr = data.qr;
                        this.whatsappConnected = data.connected;
                        if (data.connected) {
                            this.stopWhatsAppQrPolling();
                            await this.getChannelStatus();
                            this.$nextTick(() => {
                                if (window.refreshIcons) window.refreshIcons();
                            });
                        }
                    }
                } catch (e) {
                    console.error('Failed to poll WhatsApp QR', e);
                }
            },

            /**
             * Stop QR polling
             */
            stopWhatsAppQrPolling() {
                if (this.whatsappQrPolling) {
                    clearInterval(this.whatsappQrPolling);
                    this.whatsappQrPolling = null;
                }
            },

            /**
             * Get the count of running channels (for sidebar badge)
             */
            runningChannelCount() {
                return Object.values(this.channelStatus).filter(s => s.running).length;
            }
        };
    }
};
