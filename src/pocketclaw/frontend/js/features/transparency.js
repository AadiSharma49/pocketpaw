/**
 * PocketPaw - Transparency Feature Module
 *
 * Created: 2026-02-05
 * Extracted from app.js as part of componentization refactor.
 *
 * Contains transparency panel features:
 * - Identity panel
 * - Memory panel (sessions + long-term)
 * - Audit logs
 */

window.PocketPaw = window.PocketPaw || {};

window.PocketPaw.Transparency = {
    name: 'Transparency',
    /**
     * Get initial state for Transparency features
     */
    getState() {
        return {
            // Identity
            showIdentity: false,
            identityLoading: false,
            identityData: {},

            // Memory
            showMemory: false,
            memoryTab: 'sessions',  // 'sessions', 'long_term'
            sessionsList: [],       // List of all sessions
            sessionMemory: [],      // Current session history
            selectedSession: null,  // Currently selected session ID
            longTermMemory: [],
            memoryLoading: false,
            memorySearch: '',
            memoryStats: { total_memories: 0, active_context: 0, archived: 0, user_interactions: 0 },

            // Audit
            showAudit: false,
            auditLoading: false,
            auditLogs: [],

            // Activity log (for system events)
            activityLog: [],
            sessionId: null,

            // Security audit
            securityAuditResults: null,
            securityAuditLoading: false,

            // Self-audit
            selfAuditReports: [],
            selfAuditDetail: null,
            selfAuditRunning: false
        };
    },

    /**
     * Get methods for Transparency features
     */
    getMethods() {
        return {
            // ==================== Connection Info ====================

            /**
             * Handle connection info (capture session ID)
             */
            handleConnectionInfo(data) {
                this.handleNotification(data);
                if (data.id) {
                    this.sessionId = data.id;
                    this.log(`Session ID: ${data.id}`, 'info');
                }
            },

            /**
             * Handle system event (Activity Log + Mission Control events)
             */
            handleSystemEvent(data) {
                const time = Tools.formatTime();
                const eventType = data.event_type || '';

                // Handle Mission Control events
                if (eventType.startsWith('mc_')) {
                    this.handleMCEvent(data);
                    return;
                }

                // Handle live audit entries
                if (eventType === 'audit_entry') {
                    if (this.showAudit && data.data) {
                        this.auditLogs.unshift(data.data);
                    }
                    return;
                }

                // Handle standard system events
                let message = '';
                let level = 'info';

                if (eventType === 'thinking') {
                    if (data.data && data.data.content) {
                        const snippet = data.data.content.substring(0, 120);
                        const ellipsis = data.data.content.length > 120 ? '...' : '';
                        message = `<span class="text-white/40 italic">${snippet}${ellipsis}</span>`;
                    } else {
                        message = `<span class="text-accent animate-pulse">Thinking...</span>`;
                    }
                } else if (eventType === 'thinking_done') {
                    message = `<span class="text-white/40">Thinking complete</span>`;
                } else if (eventType === 'tool_start') {
                    message = `🔧 <b>${data.data.name}</b> <span class="text-white/50">${JSON.stringify(data.data.params)}</span>`;
                    level = 'warning';
                } else if (eventType === 'tool_result') {
                    const isError = data.data.status === 'error';
                    level = isError ? 'error' : 'success';
                    message = `${isError ? '❌' : '✅'} <b>${data.data.name}</b> result: <span class="text-white/50">${String(data.data.result).substring(0, 50)} ${(String(data.data.result).length > 50) ? '...' : ''}</span>`;
                } else {
                    message = `Unknown event: ${eventType}`;
                }

                this.activityLog.push({ time, message, level });

                // Also feed plain-text version into Terminal logs
                if (eventType === 'thinking') {
                    this.log('Thinking...', 'info');
                } else if (eventType === 'tool_start') {
                    const name = data.data?.name || 'unknown';
                    const params = JSON.stringify(data.data?.params || {}).substring(0, 80);
                    this.log(`[TOOL] ${name} ${params}`, 'warning');
                } else if (eventType === 'tool_result') {
                    const name = data.data?.name || 'unknown';
                    const isErr = data.data?.status === 'error';
                    const result = String(data.data?.result || '').substring(0, 80);
                    this.log(`[${isErr ? 'ERR' : 'OK'}] ${name}: ${result}`, isErr ? 'error' : 'success');
                }

                // Auto-scroll activity log
                this.$nextTick(() => {
                    const term = this.$refs.activityLog;
                    if (term) term.scrollTop = term.scrollHeight;
                });
            },

            // ==================== Identity Panel ====================

            openIdentity() {
                this.showIdentity = true;
                this.identityLoading = true;
                fetch('/api/identity')
                    .then(r => r.json())
                    .then(data => {
                        this.identityData = data;
                        this.identityLoading = false;
                    })
                    .catch(e => {
                        this.showToast('Failed to load identity', 'error');
                        this.identityLoading = false;
                    });
            },

            // ==================== Memory Panel ====================

            openMemory() {
                this.showMemory = true;
                this.memoryLoading = true;
                this.loadSessionsList();
                this.loadLongTermMemory();
            },

            loadSessionsList() {
                fetch('/api/memory/sessions')
                    .then(r => r.json())
                    .then(data => {
                        this.sessionsList = Array.isArray(data) ? data : (data.sessions || []);
                        this.updateMemoryStats();

                        // Auto-select current session if in list
                        if (this.sessionId && this.sessionsList.some(s => s.id === this.sessionId)) {
                            this.selectMemorySession(this.sessionId);
                        }
                    })
                    .catch(e => {
                        console.error('Failed to load sessions:', e);
                    });
            },

            selectMemorySession(sessionId) {
                this.selectedSession = sessionId;
                fetch(`/api/memory/session?id=${sessionId}`)
                    .then(r => r.json())
                    .then(data => {
                        this.sessionMemory = data;
                    });
            },

            loadLongTermMemory() {
                fetch('/api/memory/long_term')
                    .then(r => r.json())
                    .then(data => {
                        this.longTermMemory = data;
                        this.updateMemoryStats();
                        this.memoryLoading = false;
                    })
                    .catch(e => {
                        console.error('Failed to load memories:', e);
                        this.memoryLoading = false;
                    });
            },

            updateMemoryStats() {
                const totalSessionMessages = this.sessionsList.reduce((sum, s) => sum + (s.message_count || 0), 0);
                this.memoryStats = {
                    total_memories: this.longTermMemory.length + totalSessionMessages,
                    active_sessions: this.sessionsList.length,
                    long_term_facts: this.longTermMemory.length,
                    user_interactions: this.sessionMemory.filter(m => m.role === 'user').length
                };
            },

            /**
             * Get filtered memories based on search query
             */
            getFilteredMemories() {
                const search = this.memorySearch.toLowerCase().trim();
                const allMemories = [
                    ...this.longTermMemory.map(m => ({ ...m, type: 'long_term', id: m.timestamp })),
                    ...this.sessionMemory.map((m, i) => ({ ...m, type: 'session', id: `session-${i}`, created_at: new Date().toISOString() }))
                ];

                if (!search) return allMemories;

                return allMemories.filter(m =>
                    m.content?.toLowerCase().includes(search) ||
                    m.tags?.some(t => t.toLowerCase().includes(search))
                );
            },

            /**
             * Get CSS class for memory type badge
             */
            getMemoryTypeClass(type) {
                const classes = {
                    'long_term': 'bg-purple-500/20 text-purple-400 border-purple-500/30',
                    'session': 'bg-blue-500/20 text-blue-400 border-blue-500/30',
                    'daily': 'bg-green-500/20 text-green-400 border-green-500/30'
                };
                return classes[type] || 'bg-gray-500/20 text-gray-400 border-gray-500/30';
            },

            /**
             * Format date for display
             */
            formatDate(dateStr) {
                if (!dateStr) return '';
                try {
                    const date = new Date(dateStr);
                    return date.toLocaleDateString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                    });
                } catch (e) {
                    return dateStr;
                }
            },

            /**
             * Delete a memory (placeholder - needs backend endpoint)
             */
            deleteMemory(id) {
                fetch(`/api/memory/long_term/${encodeURIComponent(id)}`, { method: 'DELETE' })
                    .then(r => {
                        if (!r.ok) throw new Error('Delete failed');
                        this.longTermMemory = this.longTermMemory.filter(m => m.id !== id);
                        this.updateMemoryStats();
                        this.showToast('Memory deleted', 'success');
                    })
                    .catch(() => {
                        this.showToast('Failed to delete memory', 'error');
                    });
            },

            // ==================== Audit Panel ====================

            openAudit() {
                this.showAudit = true;
                this.auditLoading = true;
                fetch('/api/audit')
                    .then(r => r.json())
                    .then(data => {
                        this.auditLogs = data;
                        this.auditLoading = false;
                    })
                    .catch(e => {
                        this.showToast('Failed to load audit logs', 'error');
                        this.auditLoading = false;
                    });
            },

            // ==================== Security Audit ====================

            runSecurityAudit() {
                this.securityAuditLoading = true;
                this.securityAuditResults = null;
                fetch('/api/security-audit', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        this.securityAuditResults = data;
                        this.securityAuditLoading = false;
                    })
                    .catch(() => {
                        this.showToast('Security audit failed', 'error');
                        this.securityAuditLoading = false;
                    });
            },

            // ==================== Self-Audit ====================

            loadSelfAuditReports() {
                fetch('/api/self-audit/reports')
                    .then(r => r.json())
                    .then(data => { this.selfAuditReports = data; })
                    .catch(() => {
                        this.showToast('Failed to load self-audit reports', 'error');
                    });
            },

            viewSelfAuditReport(date) {
                fetch(`/api/self-audit/reports/${encodeURIComponent(date)}`)
                    .then(r => r.json())
                    .then(data => { this.selfAuditDetail = data; })
                    .catch(() => {
                        this.showToast('Failed to load report', 'error');
                    });
            },

            triggerSelfAudit() {
                this.selfAuditRunning = true;
                fetch('/api/self-audit/run', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        this.selfAuditDetail = data;
                        this.selfAuditRunning = false;
                        this.loadSelfAuditReports();
                        this.showToast(`Self-audit complete: ${data.passed}/${data.total_checks} passed`, 'success');
                    })
                    .catch(() => {
                        this.showToast('Self-audit failed', 'error');
                        this.selfAuditRunning = false;
                    });
            }
        };
    }
};

window.PocketPaw.Loader.register('Transparency', window.PocketPaw.Transparency);
