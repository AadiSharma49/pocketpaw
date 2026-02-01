/**
 * PocketPaw Main Application
 * Alpine.js component for the dashboard
 */

function app() {
    return {
        // View state
        view: 'chat',
        showSettings: false,
        showScreenshot: false,
        showFileBrowser: false,
        showReminders: false,
        screenshotSrc: '',

        // Reminders state
        reminders: [],
        reminderInput: '',
        reminderLoading: false,

        // Intentions state
        showIntentions: false,
        intentions: [],
        intentionLoading: false,
        intentionForm: {
            name: '',
            prompt: '',
            schedulePreset: '',
            customCron: '',
            includeSystemStatus: false
        },

        // Skills state
        showSkills: false,
        skills: [],
        skillsLoading: false,

        // File browser state
        filePath: '~',
        files: [],
        fileLoading: false,
        fileError: null,
        
        // Agent state
        agentActive: false,
        isStreaming: false,
        streamingContent: '',
        streamingMessageId: null,
        hasShownWelcome: false,
        
        // Messages
        messages: [],
        logs: [],
        inputText: '',
        
        // System status
        status: {
            cpu: '—',
            ram: '—',
            disk: '—',
            battery: '—'
        },
        
        // Settings
        settings: {
            agentBackend: 'open_interpreter',
            llmProvider: 'auto'
        },
        
        // API Keys (not persisted client-side)
        apiKeys: {
            anthropic: '',
            openai: ''
        },

        /**
         * Initialize the app
         */
        init() {
            this.log('PocketPaw Dashboard initialized', 'info');

            // Register event handlers first
            this.setupSocketHandlers();

            // Connect WebSocket (singleton - will only connect once)
            socket.connect();

            // Start status polling (low frequency)
            this.startStatusPolling();

            // Refresh Lucide icons after initial render
            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Set up WebSocket event handlers
         */
        setupSocketHandlers() {
            // Clear existing handlers to prevent duplicates
            socket.clearHandlers();
            
            const onConnected = () => {
                this.log('Connected to PocketPaw Engine', 'success');
                // Fetch initial status
                socket.runTool('status');
            };
            
            socket.on('connected', onConnected);
            
            // If already connected, trigger manually
            if (socket.isConnected) {
                onConnected();
            }
            
            socket.on('disconnected', () => {
                this.log('Disconnected from server', 'error');
            });
            
            socket.on('message', (data) => this.handleMessage(data));
            socket.on('notification', (data) => this.handleNotification(data));
            socket.on('status', (data) => this.handleStatus(data));
            socket.on('screenshot', (data) => this.handleScreenshot(data));
            socket.on('code', (data) => this.handleCode(data));
            socket.on('error', (data) => this.handleError(data));
            socket.on('stream_start', () => this.startStreaming());
            socket.on('stream_end', () => this.endStreaming());
            socket.on('files', (data) => this.handleFiles(data));

            // Reminder handlers
            socket.on('reminders', (data) => this.handleReminders(data));
            socket.on('reminder_added', (data) => this.handleReminderAdded(data));
            socket.on('reminder_deleted', (data) => this.handleReminderDeleted(data));
            socket.on('reminder', (data) => this.handleReminderTriggered(data));

            // Intention handlers
            socket.on('intentions', (data) => this.handleIntentions(data));
            socket.on('intention_created', (data) => this.handleIntentionCreated(data));
            socket.on('intention_updated', (data) => this.handleIntentionUpdated(data));
            socket.on('intention_toggled', (data) => this.handleIntentionToggled(data));
            socket.on('intention_deleted', (data) => this.handleIntentionDeleted(data));
            socket.on('intention_event', (data) => this.handleIntentionEvent(data));

            // Skills handlers
            socket.on('skills', (data) => this.handleSkills(data));
            socket.on('skill_started', (data) => this.handleSkillStarted(data));
            socket.on('skill_completed', (data) => this.handleSkillCompleted(data));
            socket.on('skill_error', (data) => this.handleSkillError(data));
        },

        /**
         * Handle notification
         */
        handleNotification(data) {
            const content = data.content || '';
            
            // Skip duplicate connection messages
            if (content.includes('Connected to PocketPaw') && this.hasShownWelcome) {
                return;
            }
            if (content.includes('Connected to PocketPaw')) {
                this.hasShownWelcome = true;
            }
            
            this.showToast(content, 'info');
            this.log(content, 'info');
        },

        /**
         * Handle incoming message
         */
        handleMessage(data) {
            const content = data.content || '';
            
            // Check if it's a status update (don't show in chat)
            if (content.includes('System Status') || content.includes('🧠 CPU:')) {
                this.status = Tools.parseStatus(content);
                return;
            }
            
            // Handle streaming vs complete messages
            if (this.isStreaming) {
                this.streamingContent += content;
            } else {
                this.addMessage('assistant', content);
            }
            
            this.log(content.substring(0, 80) + (content.length > 80 ? '...' : ''), 'info');
        },

        /**
         * Handle status updates
         */
        handleStatus(data) {
            if (data.content) {
                this.status = Tools.parseStatus(data.content);
            }
        },

        /**
         * Handle screenshot
         */
        handleScreenshot(data) {
            if (data.image) {
                this.screenshotSrc = `data:image/png;base64,${data.image}`;
                this.showScreenshot = true;
            }
        },

        /**
         * Handle code blocks
         */
        handleCode(data) {
            const content = data.content || '';
            if (this.isStreaming) {
                this.streamingContent += '\n```\n' + content + '\n```\n';
            } else {
                this.addMessage('assistant', '```\n' + content + '\n```');
            }
        },

        /**
         * Handle errors
         */
        handleError(data) {
            const content = data.content || 'Unknown error';
            this.addMessage('assistant', '❌ ' + content);
            this.log(content, 'error');
            this.showToast(content, 'error');
            this.endStreaming();

            // If file browser is open, show error there
            if (this.showFileBrowser) {
                this.fileLoading = false;
                this.fileError = content;
            }
        },

        /**
         * Handle file browser data
         */
        handleFiles(data) {
            this.fileLoading = false;
            this.fileError = null;

            if (data.error) {
                this.fileError = data.error;
                return;
            }

            this.filePath = data.path || '~';
            this.files = data.files || [];

            // Refresh Lucide icons after Alpine renders
            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Handle reminders list
         */
        handleReminders(data) {
            this.reminders = data.reminders || [];
            this.reminderLoading = false;
        },

        /**
         * Handle reminder added
         */
        handleReminderAdded(data) {
            this.reminders.push(data.reminder);
            this.reminderInput = '';
            this.reminderLoading = false;
            this.showToast('Reminder set!', 'success');
        },

        /**
         * Handle reminder deleted
         */
        handleReminderDeleted(data) {
            this.reminders = this.reminders.filter(r => r.id !== data.id);
        },

        /**
         * Handle reminder triggered (notification)
         */
        handleReminderTriggered(data) {
            const reminder = data.reminder;
            this.showToast(`Reminder: ${reminder.text}`, 'info');
            this.addMessage('assistant', `Reminder: ${reminder.text}`);

            // Remove from local list
            this.reminders = this.reminders.filter(r => r.id !== reminder.id);

            // Try desktop notification
            if (Notification.permission === 'granted') {
                new Notification('PocketPaw Reminder', {
                    body: reminder.text,
                    icon: '/static/icon.png'
                });
            }
        },

        /**
         * Start streaming mode
         */
        startStreaming() {
            this.isStreaming = true;
            this.streamingContent = '';
        },

        /**
         * End streaming mode
         */
        endStreaming() {
            if (this.isStreaming && this.streamingContent) {
                this.addMessage('assistant', this.streamingContent);
            }
            this.isStreaming = false;
            this.streamingContent = '';
        },

        /**
         * Add a message to the chat
         */
        addMessage(role, content) {
            this.messages.push({
                role,
                content,
                time: Tools.formatTime()
            });
            
            // Auto scroll to bottom
            this.$nextTick(() => {
                if (this.$refs.messages) {
                    this.$refs.messages.scrollTop = this.$refs.messages.scrollHeight;
                }
            });
        },

        /**
         * Send a chat message
         */
        sendMessage() {
            const text = this.inputText.trim();
            if (!text) return;

            // Check for skill command (starts with /)
            if (text.startsWith('/')) {
                const parts = text.slice(1).split(' ');
                const skillName = parts[0];
                const args = parts.slice(1).join(' ');

                // Add user message
                this.addMessage('user', text);
                this.inputText = '';

                // Run the skill
                socket.send('run_skill', { name: skillName, args });
                this.log(`Running skill: /${skillName} ${args}`, 'info');
                return;
            }

            // Add user message
            this.addMessage('user', text);
            this.inputText = '';

            // Start streaming indicator
            this.startStreaming();

            // Send to server
            socket.chat(text);

            this.log(`You: ${text}`, 'info');
        },

        /**
         * Run a tool
         */
        runTool(tool) {
            this.log(`Running tool: ${tool}`, 'info');

            // Special handling for file browser
            if (tool === 'fetch') {
                this.openFileBrowser();
                return;
            }

            socket.runTool(tool);
        },

        /**
         * Open file browser modal
         */
        openFileBrowser() {
            this.showFileBrowser = true;
            this.fileLoading = true;
            this.fileError = null;
            this.files = [];
            this.filePath = '~';

            // Refresh icons after modal renders
            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });

            socket.send('browse', { path: '~' });
        },

        /**
         * Navigate to a directory
         */
        navigateTo(path) {
            this.fileLoading = true;
            this.fileError = null;
            socket.send('browse', { path });
        },

        /**
         * Navigate up one directory
         */
        navigateUp() {
            const parts = this.filePath.split('/').filter(s => s);
            parts.pop();
            const newPath = parts.length > 0 ? parts.join('/') : '~';
            this.navigateTo(newPath);
        },

        /**
         * Navigate to a path segment (breadcrumb click)
         */
        navigateToSegment(index) {
            const parts = this.filePath.split('/').filter(s => s);
            const newPath = parts.slice(0, index + 1).join('/');
            this.navigateTo(newPath || '~');
        },

        /**
         * Select a file or folder
         */
        selectFile(item) {
            if (item.isDir) {
                // Navigate into directory
                const newPath = this.filePath === '~'
                    ? item.name
                    : `${this.filePath}/${item.name}`;
                this.navigateTo(newPath);
            } else {
                // File selected - could download or preview
                this.log(`Selected file: ${item.name}`, 'info');
                this.showToast(`Selected: ${item.name}`, 'info');
                // TODO: Add file download/preview functionality
            }
        },

        /**
         * Open reminders panel
         */
        openReminders() {
            this.showReminders = true;
            this.reminderLoading = true;
            socket.send('get_reminders');

            // Request notification permission
            if (Notification.permission === 'default') {
                Notification.requestPermission();
            }

            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Add a reminder
         */
        addReminder() {
            const text = this.reminderInput.trim();
            if (!text) return;

            this.reminderLoading = true;
            socket.send('add_reminder', { message: text });
            this.log(`Setting reminder: ${text}`, 'info');
        },

        /**
         * Delete a reminder
         */
        deleteReminder(id) {
            socket.send('delete_reminder', { id });
        },

        /**
         * Format reminder time for display
         */
        formatReminderTime(reminder) {
            const date = new Date(reminder.trigger_at);
            return date.toLocaleString(undefined, {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        },

        // ==================== Intentions ====================

        /**
         * Handle intentions list
         */
        handleIntentions(data) {
            this.intentions = data.intentions || [];
            this.intentionLoading = false;
        },

        /**
         * Handle intention created
         */
        handleIntentionCreated(data) {
            this.intentions.push(data.intention);
            this.resetIntentionForm();
            this.showToast('Intention created!', 'success');
            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Handle intention updated
         */
        handleIntentionUpdated(data) {
            const index = this.intentions.findIndex(i => i.id === data.intention.id);
            if (index !== -1) {
                this.intentions[index] = data.intention;
            }
        },

        /**
         * Handle intention toggled
         */
        handleIntentionToggled(data) {
            const index = this.intentions.findIndex(i => i.id === data.intention.id);
            if (index !== -1) {
                this.intentions[index] = data.intention;
            }
            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Handle intention deleted
         */
        handleIntentionDeleted(data) {
            this.intentions = this.intentions.filter(i => i.id !== data.id);
        },

        /**
         * Handle intention execution events
         */
        handleIntentionEvent(data) {
            const eventType = data.type;

            if (eventType === 'intention_started') {
                this.showToast(`Running: ${data.intention_name}`, 'info');
                this.log(`Intention started: ${data.intention_name}`, 'info');
                this.startStreaming();
            } else if (eventType === 'intention_completed') {
                this.log(`Intention completed: ${data.intention_name}`, 'success');
                this.endStreaming();
                // Refresh intentions to update next_run time
                socket.send('get_intentions');
            } else if (eventType === 'intention_error') {
                this.showToast(`Error: ${data.error}`, 'error');
                this.log(`Intention error: ${data.error}`, 'error');
                this.endStreaming();
            } else if (data.content) {
                // Stream content from agent
                if (this.isStreaming) {
                    this.streamingContent += data.content;
                }
            }
        },

        /**
         * Open intentions panel
         */
        openIntentions() {
            this.showIntentions = true;
            this.intentionLoading = true;
            socket.send('get_intentions');

            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Create a new intention
         */
        createIntention() {
            const { name, prompt, schedulePreset, customCron, includeSystemStatus } = this.intentionForm;

            if (!name.trim() || !prompt.trim() || !schedulePreset) {
                this.showToast('Please fill in all fields', 'error');
                return;
            }

            const schedule = schedulePreset === 'custom' ? customCron : schedulePreset;
            if (!schedule) {
                this.showToast('Please enter a schedule', 'error');
                return;
            }

            const contextSources = includeSystemStatus ? ['system_status', 'datetime'] : ['datetime'];

            socket.send('create_intention', {
                name: name.trim(),
                prompt: prompt.trim(),
                trigger: { type: 'cron', schedule },
                context_sources: contextSources,
                enabled: true
            });

            this.log(`Creating intention: ${name}`, 'info');
        },

        /**
         * Toggle intention enabled state
         */
        toggleIntention(id) {
            socket.send('toggle_intention', { id });
        },

        /**
         * Delete an intention
         */
        deleteIntention(id) {
            if (confirm('Delete this intention?')) {
                socket.send('delete_intention', { id });
            }
        },

        /**
         * Run an intention immediately
         */
        runIntention(id) {
            socket.send('run_intention', { id });
        },

        /**
         * Reset intention form
         */
        resetIntentionForm() {
            this.intentionForm = {
                name: '',
                prompt: '',
                schedulePreset: '',
                customCron: '',
                includeSystemStatus: false
            };
        },

        /**
         * Format next run time for display
         */
        formatNextRun(isoString) {
            if (!isoString) return '';
            const date = new Date(isoString);
            const now = new Date();
            const diff = date - now;

            // If less than 1 hour away, show relative time
            if (diff > 0 && diff < 3600000) {
                const mins = Math.round(diff / 60000);
                return `in ${mins}m`;
            }

            // Otherwise show time
            return date.toLocaleString(undefined, {
                hour: '2-digit',
                minute: '2-digit'
            });
        },

        // ==================== Skills ====================

        /**
         * Handle skills list
         */
        handleSkills(data) {
            this.skills = data.skills || [];
            this.skillsLoading = false;
            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Handle skill started
         */
        handleSkillStarted(data) {
            this.showToast(`Running: ${data.skill_name}`, 'info');
            this.log(`Skill started: ${data.skill_name}`, 'info');
        },

        /**
         * Handle skill completed
         */
        handleSkillCompleted(data) {
            this.log(`Skill completed: ${data.skill_name}`, 'success');
        },

        /**
         * Handle skill error
         */
        handleSkillError(data) {
            this.showToast(`Skill error: ${data.error}`, 'error');
            this.log(`Skill error: ${data.error}`, 'error');
        },

        /**
         * Open skills panel
         */
        openSkills() {
            this.showSkills = true;
            this.skillsLoading = true;
            socket.send('get_skills');

            this.$nextTick(() => {
                if (window.refreshIcons) window.refreshIcons();
            });
        },

        /**
         * Run a skill
         */
        runSkill(name, args = '') {
            this.showSkills = false;
            socket.send('run_skill', { name, args });
            this.log(`Running skill: ${name} ${args}`, 'info');
        },

        /**
         * Check if input is a skill command and run it
         */
        checkSkillCommand(text) {
            if (text.startsWith('/')) {
                const parts = text.slice(1).split(' ');
                const skillName = parts[0];
                const args = parts.slice(1).join(' ');

                // Check if skill exists
                const skill = this.skills.find(s => s.name === skillName);
                if (skill) {
                    this.runSkill(skillName, args);
                    return true;
                }
            }
            return false;
        },

        /**
         * Toggle agent mode
         */
        toggleAgent() {
            socket.toggleAgent(this.agentActive);
            this.log(`Switched Agent Mode: ${this.agentActive ? 'ON' : 'OFF'}`, 'info');
        },

        /**
         * Save settings
         */
        saveSettings() {
            socket.saveSettings(this.settings.agentBackend, this.settings.llmProvider);
            this.log('Settings updated', 'info');
            this.showToast('Settings saved', 'success');
        },

        /**
         * Save API key
         */
        saveApiKey(provider) {
            const key = this.apiKeys[provider];
            if (!key) {
                this.showToast('Please enter an API key', 'error');
                return;
            }
            
            socket.saveApiKey(provider, key);
            this.apiKeys[provider] = ''; // Clear input
            this.log(`Saved ${provider} API key`, 'success');
            this.showToast(`${provider.charAt(0).toUpperCase() + provider.slice(1)} API key saved!`, 'success');
        },

        /**
         * Start polling for system status (every 10 seconds, only when connected)
         */
        startStatusPolling() {
            setInterval(() => {
                if (socket.isConnected) {
                    socket.runTool('status');
                }
            }, 10000); // Poll every 10 seconds, not 3
        },

        /**
         * Add log entry
         */
        log(message, level = 'info') {
            this.logs.push({
                time: Tools.formatTime(),
                message,
                level
            });
            
            // Keep only last 100 logs
            if (this.logs.length > 100) {
                this.logs.shift();
            }
            
            // Auto scroll terminal
            this.$nextTick(() => {
                if (this.$refs.terminal) {
                    this.$refs.terminal.scrollTop = this.$refs.terminal.scrollHeight;
                }
            });
        },

        /**
         * Format message content
         */
        formatMessage(content) {
            return Tools.formatMessage(content);
        },

        /**
         * Get current time string
         */
        currentTime() {
            return Tools.formatTime();
        },

        /**
         * Show toast notification
         */
        showToast(message, type = 'info') {
            Tools.showToast(message, type, this.$refs.toasts);
        }
    };
}
