/**
 * PocketPaw - Mission Control Feature Module
 *
 * Created: 2026-02-05
 * Extracted from app.js as part of componentization refactor.
 *
 * Contains all Crew (Mission Control) related state and methods:
 * - Agent CRUD operations
 * - Task CRUD operations
 * - Task execution (run/stop)
 * - WebSocket event handling for real-time updates
 * - Agent Activity Sheet
 * - Comments/Thread
 * - Deliverables
 */

window.PocketPaw = window.PocketPaw || {};

window.PocketPaw.MissionControl = {
    name: 'MissionControl',
    /**
     * Get initial state for Mission Control
     */
    getState() {
        return {
            missionControl: {
                loading: false,
                taskFilter: 'all',
                agents: [],
                tasks: [],
                activities: [],
                stats: { total_agents: 0, active_tasks: 0, completed_today: 0, total_documents: 0 },
                selectedTask: null,
                showCreateAgent: false,
                showCreateTask: false,
                agentForm: { name: '', role: '', description: '', specialties: '' },
                taskForm: { title: '', description: '', priority: 'medium', assignee: '', tags: '' },
                // Task execution state
                runningTasks: {},  // {task_id: {agentName, agentId, taskTitle, output: [], startedAt}}
                liveOutput: '',    // Current live output for selected task
                // Agent Activity Sheet state
                showAgentActivitySheet: false,
                activeAgentTask: null,  // {taskId, agentId, agentName, taskTitle}
                // Comments/Thread state
                taskMessages: [],
                messageInput: '',
                messagesLoading: false,
                // Deliverables state
                taskDeliverables: [],
                deliverablesLoading: false,
            }
        };
    },

    /**
     * Get methods for Mission Control
     * Note: 'this' will be bound to the Alpine component
     */
    getMethods() {
        return {
            // ==================== Mission Control Data Loading ====================

            /**
             * Load Mission Control data from API
             */
            async loadMCData() {
                // Skip if already loaded and not stale
                if (this.missionControl.agents.length > 0 && !this.missionControl.loading) {
                    // Just refresh activity feed
                    try {
                        const activityRes = await fetch('/api/mission-control/activity');
                        if (activityRes.ok) {
                            const data = await activityRes.json();
                            this.missionControl.activities = data.activities || [];
                        }
                    } catch (e) { /* ignore */ }
                    this.$nextTick(() => { if (window.refreshIcons) window.refreshIcons(); });
                    return;
                }

                this.missionControl.loading = true;
                try {
                    const [agentsRes, tasksRes, activityRes, statsRes] = await Promise.all([
                        fetch('/api/mission-control/agents'),
                        fetch('/api/mission-control/tasks'),
                        fetch('/api/mission-control/activity'),
                        fetch('/api/mission-control/stats')
                    ]);

                    // Unwrap API responses (backend returns {agents: [...], count: N} format)
                    if (agentsRes.ok) {
                        const data = await agentsRes.json();
                        this.missionControl.agents = data.agents || [];
                    }
                    if (tasksRes.ok) {
                        const data = await tasksRes.json();
                        this.missionControl.tasks = data.tasks || [];
                    }
                    if (activityRes.ok) {
                        const data = await activityRes.json();
                        this.missionControl.activities = data.activities || [];
                    }
                    if (statsRes.ok) {
                        const data = await statsRes.json();
                        const raw = data.stats || data;
                        // Map backend stats to frontend format
                        this.missionControl.stats = {
                            total_agents: raw.agents?.total || 0,
                            active_tasks: (raw.tasks?.by_status?.in_progress || 0) + (raw.tasks?.by_status?.assigned || 0),
                            completed_today: raw.tasks?.by_status?.done || 0,
                            total_documents: raw.documents?.total || 0
                        };
                    }
                } catch (e) {
                    console.error('Failed to load Crew data:', e);
                    this.showToast('Failed to load Crew', 'error');
                } finally {
                    this.missionControl.loading = false;
                }
            },

            /**
             * Get filtered tasks based on current filter
             */
            getFilteredMCTasks() {
                const filter = this.missionControl.taskFilter;
                if (filter === 'all') return this.missionControl.tasks;
                return this.missionControl.tasks.filter(t => t.status === filter);
            },

            // ==================== Agent CRUD ====================

            /**
             * Create a new agent
             */
            async createMCAgent() {
                const form = this.missionControl.agentForm;
                if (!form.name || !form.role) return;

                try {
                    const specialties = form.specialties
                        ? form.specialties.split(',').map(s => s.trim()).filter(s => s)
                        : [];

                    const res = await fetch('/api/mission-control/agents', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            name: form.name,
                            role: form.role,
                            description: form.description,
                            specialties: specialties
                        })
                    });

                    if (res.ok) {
                        const data = await res.json();
                        const agent = data.agent || data;  // Unwrap if wrapped
                        this.missionControl.agents.push(agent);
                        this.missionControl.stats.total_agents++;
                        this.missionControl.showCreateAgent = false;
                        this.missionControl.agentForm = { name: '', role: '', description: '', specialties: '' };
                        this.showToast('Agent created!', 'success');
                        this.$nextTick(() => {
                            if (window.refreshIcons) window.refreshIcons();
                        });
                    } else {
                        const err = await res.json();
                        this.showToast(err.detail || 'Failed to create agent', 'error');
                    }
                } catch (e) {
                    console.error('Failed to create agent:', e);
                    this.showToast('Failed to create agent', 'error');
                }
            },

            /**
             * Delete an agent
             */
            async deleteMCAgent(agentId) {
                if (!confirm('Delete this agent?')) return;

                try {
                    const res = await fetch(`/api/mission-control/agents/${agentId}`, {
                        method: 'DELETE'
                    });

                    if (res.ok) {
                        this.missionControl.agents = this.missionControl.agents.filter(a => a.id !== agentId);
                        this.missionControl.stats.total_agents--;
                        this.showToast('Agent deleted', 'info');
                    }
                } catch (e) {
                    console.error('Failed to delete agent:', e);
                    this.showToast('Failed to delete agent', 'error');
                }
            },

            // ==================== Task CRUD ====================

            /**
             * Create a new task
             */
            async createMCTask() {
                const form = this.missionControl.taskForm;
                if (!form.title) return;

                try {
                    const tags = form.tags
                        ? form.tags.split(',').map(s => s.trim()).filter(s => s)
                        : [];

                    const body = {
                        title: form.title,
                        description: form.description,
                        priority: form.priority,
                        tags: tags
                    };

                    if (form.assignee) {
                        body.assignee_ids = [form.assignee];
                    }

                    const res = await fetch('/api/mission-control/tasks', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });

                    if (res.ok) {
                        const data = await res.json();
                        const task = data.task || data;  // Unwrap if wrapped
                        this.missionControl.tasks.unshift(task);
                        this.missionControl.stats.active_tasks++;
                        this.missionControl.showCreateTask = false;
                        this.missionControl.taskForm = { title: '', description: '', priority: 'medium', assignee: '', tags: '' };
                        this.showToast('Task created!', 'success');
                        // Reload activity feed
                        const activityRes = await fetch('/api/mission-control/activity');
                        if (activityRes.ok) {
                            const actData = await activityRes.json();
                            this.missionControl.activities = actData.activities || [];
                        }
                        this.$nextTick(() => {
                            if (window.refreshIcons) window.refreshIcons();
                        });
                    } else {
                        const err = await res.json();
                        this.showToast(err.detail || 'Failed to create task', 'error');
                    }
                } catch (e) {
                    console.error('Failed to create task:', e);
                    this.showToast('Failed to create task', 'error');
                }
            },

            /**
             * Delete a task
             */
            async deleteMCTask(taskId) {
                if (!confirm('Delete this task?')) return;

                try {
                    const res = await fetch(`/api/mission-control/tasks/${taskId}`, {
                        method: 'DELETE'
                    });

                    if (res.ok) {
                        this.missionControl.tasks = this.missionControl.tasks.filter(t => t.id !== taskId);
                        this.missionControl.stats.active_tasks = Math.max(0, this.missionControl.stats.active_tasks - 1);
                        this.showToast('Task deleted', 'info');
                    }
                } catch (e) {
                    console.error('Failed to delete task:', e);
                    this.showToast('Failed to delete task', 'error');
                }
            },

            /**
             * Select a task to show details
             */
            selectMCTask(task) {
                this.missionControl.selectedTask = task;
                this.$nextTick(() => { if (window.refreshIcons) window.refreshIcons(); });
            },

            /**
             * Update task status
             */
            async updateMCTaskStatus(taskId, status) {
                try {
                    const res = await fetch(`/api/mission-control/tasks/${taskId}/status`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ status })
                    });

                    if (res.ok) {
                        // Update local state
                        const task = this.missionControl.tasks.find(t => t.id === taskId);
                        if (task) task.status = status;
                        this.showToast(`Status updated to ${status}`, 'success');
                        // Reload activity
                        const activityRes = await fetch('/api/mission-control/activity');
                        if (activityRes.ok) {
                            const data = await activityRes.json();
                            this.missionControl.activities = data.activities || [];
                        }
                    }
                } catch (e) {
                    console.error('Failed to update task status:', e);
                    this.showToast('Failed to update status', 'error');
                }
            },

            /**
             * Update task priority
             */
            async updateMCTaskPriority(taskId, priority) {
                try {
                    const res = await fetch(`/api/mission-control/tasks/${taskId}`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ priority })
                    });

                    if (res.ok) {
                        // Update local state
                        const task = this.missionControl.tasks.find(t => t.id === taskId);
                        if (task) task.priority = priority;
                        if (this.missionControl.selectedTask?.id === taskId) {
                            this.missionControl.selectedTask.priority = priority;
                        }
                    }
                } catch (e) {
                    console.error('Failed to update task priority:', e);
                }
            },

            // ==================== Agent Helpers ====================

            /**
             * Get agent initial for avatar
             */
            getAgentInitial(agentId) {
                const agent = this.missionControl.agents.find(a => a.id === agentId);
                return agent ? agent.name.charAt(0).toUpperCase() : '?';
            },

            /**
             * Get agent name by ID
             */
            getAgentName(agentId) {
                const agent = this.missionControl.agents.find(a => a.id === agentId);
                return agent ? agent.name : 'Unknown';
            },

            /**
             * Get full agent object by ID
             */
            getAgentById(agentId) {
                return this.missionControl.agents.find(a => a.id === agentId);
            },

            /**
             * Get agents not already assigned to a task
             */
            getAvailableAgentsForTask(task) {
                if (!task) return this.missionControl.agents;
                const assignedIds = task.assignee_ids || [];
                return this.missionControl.agents.filter(a => !assignedIds.includes(a.id));
            },

            // ==================== Task Assignment ====================

            /**
             * Assign an agent to a task
             */
            async assignAgentToTask(taskId, agentId) {
                if (!taskId || !agentId) return;

                try {
                    const res = await fetch(`/api/mission-control/tasks/${taskId}/assign`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ agent_ids: [agentId] })
                    });

                    if (res.ok) {
                        const data = await res.json();
                        // Update local task state
                        const task = this.missionControl.tasks.find(t => t.id === taskId);
                        if (task && data.task) {
                            task.assignee_ids = data.task.assignee_ids;
                            task.status = data.task.status;
                        }
                        // Update selected task if it's the same
                        if (this.missionControl.selectedTask?.id === taskId && data.task) {
                            this.missionControl.selectedTask.assignee_ids = data.task.assignee_ids;
                            this.missionControl.selectedTask.status = data.task.status;
                        }
                        this.showToast('Agent assigned', 'success');
                        this.$nextTick(() => { if (window.refreshIcons) window.refreshIcons(); });
                    } else {
                        const err = await res.json();
                        this.showToast(err.detail || 'Failed to assign agent', 'error');
                    }
                } catch (e) {
                    console.error('Failed to assign agent:', e);
                    this.showToast('Failed to assign agent', 'error');
                }
            },

            /**
             * Remove an agent from a task
             */
            async unassignAgentFromTask(taskId, agentId) {
                if (!taskId || !agentId) return;

                try {
                    // Get current assignees and remove this one
                    const task = this.missionControl.tasks.find(t => t.id === taskId);
                    if (!task) return;

                    const newAssignees = (task.assignee_ids || []).filter(id => id !== agentId);

                    const res = await fetch(`/api/mission-control/tasks/${taskId}/assign`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ agent_ids: newAssignees })
                    });

                    if (res.ok) {
                        const data = await res.json();
                        // Update local task state
                        if (data.task) {
                            task.assignee_ids = data.task.assignee_ids;
                            task.status = data.task.status;
                        }
                        // Update selected task if it's the same
                        if (this.missionControl.selectedTask?.id === taskId && data.task) {
                            this.missionControl.selectedTask.assignee_ids = data.task.assignee_ids;
                            this.missionControl.selectedTask.status = data.task.status;
                        }
                        this.showToast('Agent removed', 'info');
                        this.$nextTick(() => { if (window.refreshIcons) window.refreshIcons(); });
                    } else {
                        const err = await res.json();
                        this.showToast(err.detail || 'Failed to remove agent', 'error');
                    }
                } catch (e) {
                    console.error('Failed to remove agent:', e);
                    this.showToast('Failed to remove agent', 'error');
                }
            },

            // ==================== Date Formatting ====================

            /**
             * Format date for Mission Control display
             */
            formatMCDate(dateStr) {
                if (!dateStr) return '';
                try {
                    const date = new Date(dateStr);
                    const now = new Date();
                    const diff = now - date;

                    // Less than 1 minute ago
                    if (diff < 60000) return 'Just now';
                    // Less than 1 hour ago
                    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
                    // Less than 24 hours ago
                    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
                    // Otherwise show date
                    return date.toLocaleDateString('en-US', {
                        month: 'short',
                        day: 'numeric'
                    });
                } catch (e) {
                    return dateStr;
                }
            },

            // ==================== Task Execution ====================

            /**
             * Run a task with an assigned agent
             */
            async runMCTask(taskId, agentId) {
                if (!taskId || !agentId) {
                    this.showToast('Task must have an assigned agent', 'error');
                    return;
                }

                // Get task and agent info for immediate UI update
                const task = this.missionControl.tasks.find(t => t.id === taskId);
                const agent = this.missionControl.agents.find(a => a.id === agentId);

                if (!task || !agent) {
                    this.showToast('Task or agent not found', 'error');
                    return;
                }

                try {
                    const res = await fetch(`/api/mission-control/tasks/${taskId}/run`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ agent_id: agentId })
                    });

                    if (res.ok) {
                        const data = await res.json();
                        this.showToast(data.message || 'Task started', 'success');

                        // IMMEDIATELY update local state (don't wait for WebSocket)
                        // Track as running task
                        this.missionControl.runningTasks[taskId] = {
                            agentId: agentId,
                            agentName: agent.name,
                            taskTitle: task.title,
                            output: [],
                            startedAt: new Date()
                        };

                        // Update task status locally
                        task.status = 'in_progress';
                        task.started_at = new Date().toISOString();
                        if (this.missionControl.selectedTask?.id === taskId) {
                            this.missionControl.selectedTask.status = 'in_progress';
                            this.missionControl.selectedTask.started_at = task.started_at;
                        }

                        // Update agent status locally
                        agent.status = 'active';
                        agent.current_task_id = taskId;

                        // Update stats
                        this.missionControl.stats.active_tasks++;

                        // Clear and initialize live output
                        this.missionControl.liveOutput = `Starting task with ${agent.name}...\n\n`;

                        // Refresh icons
                        this.$nextTick(() => {
                            if (window.refreshIcons) window.refreshIcons();
                        });
                    } else {
                        const err = await res.json();
                        this.showToast(err.detail || 'Failed to start task', 'error');
                    }
                } catch (e) {
                    console.error('Failed to run task:', e);
                    this.showToast('Failed to start task', 'error');
                }
            },

            /**
             * Stop a running task
             */
            async stopMCTask(taskId) {
                try {
                    const res = await fetch(`/api/mission-control/tasks/${taskId}/stop`, {
                        method: 'POST'
                    });

                    if (res.ok) {
                        this.showToast('Task stopped', 'info');

                        // Immediately update local state
                        const runningData = this.missionControl.runningTasks[taskId];
                        if (runningData) {
                            // Update agent status
                            const agent = this.missionControl.agents.find(a => a.id === runningData.agentId);
                            if (agent) {
                                agent.status = 'idle';
                                agent.current_task_id = null;
                            }
                        }

                        // Remove from running tasks
                        delete this.missionControl.runningTasks[taskId];

                        // Update task status
                        const task = this.missionControl.tasks.find(t => t.id === taskId);
                        if (task) {
                            task.status = 'blocked';
                        }
                        if (this.missionControl.selectedTask?.id === taskId) {
                            this.missionControl.selectedTask.status = 'blocked';
                        }

                        // Update stats
                        this.missionControl.stats.active_tasks = Math.max(0, this.missionControl.stats.active_tasks - 1);

                        // Close activity sheet if open for this task
                        if (this.missionControl.activeAgentTask?.taskId === taskId) {
                            this.closeAgentActivitySheet();
                        }

                        // Refresh icons
                        this.$nextTick(() => {
                            if (window.refreshIcons) window.refreshIcons();
                        });
                    } else {
                        const err = await res.json();
                        this.showToast(err.detail || 'Failed to stop task', 'error');
                    }
                } catch (e) {
                    console.error('Failed to stop task:', e);
                    this.showToast('Failed to stop task', 'error');
                }
            },

            /**
             * Check if a task is currently running
             */
            isMCTaskRunning(taskId) {
                return taskId in this.missionControl.runningTasks;
            },

            /**
             * Get live output for the selected task
             */
            getMCLiveOutput() {
                return this.missionControl.liveOutput;
            },

            // ==================== WebSocket Event Handling ====================

            /**
             * Handle Mission Control WebSocket events
             */
            handleMCEvent(data) {
                const eventType = data.event_type;
                const eventData = data.data || {};

                if (eventType === 'mc_task_started') {
                    // Task execution started
                    const taskId = eventData.task_id;
                    const agentId = eventData.agent_id;
                    const agentName = eventData.agent_name;
                    const taskTitle = eventData.task_title;

                    // Track running task
                    this.missionControl.runningTasks[taskId] = {
                        agentId: agentId,
                        agentName: agentName,
                        taskTitle: taskTitle,
                        output: [],
                        startedAt: new Date()
                    };

                    // Update task status in local state
                    const task = this.missionControl.tasks.find(t => t.id === taskId);
                    if (task) {
                        task.status = 'in_progress';
                    }

                    // Update agent status
                    const agent = this.missionControl.agents.find(a => a.id === agentId);
                    if (agent) {
                        agent.status = 'active';
                        agent.current_task_id = taskId;
                    }

                    // If this task is selected, clear the live output
                    if (this.missionControl.selectedTask?.id === taskId) {
                        this.missionControl.liveOutput = '';
                    }

                    this.showToast(`${agentName} started: ${taskTitle}`, 'info');
                    this.log(`Task started: ${taskTitle}`, 'info');

                } else if (eventType === 'mc_task_output') {
                    // Agent produced output
                    const taskId = eventData.task_id;
                    const content = eventData.content || '';
                    const outputType = eventData.output_type;

                    // Add to running task output
                    if (this.missionControl.runningTasks[taskId]) {
                        this.missionControl.runningTasks[taskId].output.push({
                            content,
                            type: outputType,
                            timestamp: new Date()
                        });
                    }

                    // If this task is selected, append to live output
                    if (this.missionControl.selectedTask?.id === taskId) {
                        if (outputType === 'message') {
                            this.missionControl.liveOutput += content;
                        } else if (outputType === 'tool_use') {
                            this.missionControl.liveOutput += `\n🔧 ${content}\n`;
                        } else if (outputType === 'tool_result') {
                            this.missionControl.liveOutput += `\n✅ ${content}\n`;
                        }

                        // Scroll live output panel
                        this.$nextTick(() => {
                            const panel = this.$refs.liveOutputPanel;
                            if (panel) panel.scrollTop = panel.scrollHeight;
                        });
                    }

                    // If Agent Activity Sheet is open for this task, scroll it too
                    if (this.missionControl.showAgentActivitySheet &&
                        this.missionControl.activeAgentTask?.taskId === taskId) {
                        this.$nextTick(() => {
                            const panel = this.$refs.agentActivityOutput;
                            if (panel) panel.scrollTop = panel.scrollHeight;
                        });
                    }

                } else if (eventType === 'mc_task_completed') {
                    // Task execution completed
                    const taskId = eventData.task_id;
                    const status = eventData.status;  // 'completed', 'error', 'stopped'
                    const error = eventData.error;

                    // Remove from running tasks
                    delete this.missionControl.runningTasks[taskId];

                    // Update task status
                    const task = this.missionControl.tasks.find(t => t.id === taskId);
                    if (task) {
                        task.status = status === 'completed' ? 'done' : 'blocked';
                        if (status === 'completed') {
                            task.completed_at = new Date().toISOString();
                        }
                    }

                    // Update agent status
                    const agentId = eventData.agent_id;
                    const agent = this.missionControl.agents.find(a => a.id === agentId);
                    if (agent) {
                        agent.status = 'idle';
                        agent.current_task_id = null;
                    }

                    // Update stats
                    if (status === 'completed') {
                        this.missionControl.stats.completed_today++;
                        this.missionControl.stats.active_tasks = Math.max(0, this.missionControl.stats.active_tasks - 1);
                    }

                    // Show notification
                    if (status === 'completed') {
                        this.showToast(`Task completed: ${task?.title || taskId}`, 'success');
                    } else if (status === 'error') {
                        this.showToast(`Task failed: ${error || 'Unknown error'}`, 'error');
                    } else if (status === 'stopped') {
                        this.showToast('Task stopped', 'info');
                    }

                    this.log(`Task ${status}: ${task?.title || taskId}`, status === 'completed' ? 'success' : 'error');

                    // Refresh icons
                    this.$nextTick(() => {
                        if (window.refreshIcons) window.refreshIcons();
                    });

                } else if (eventType === 'mc_activity_created') {
                    // New activity logged
                    const activity = eventData.activity;
                    if (activity) {
                        // Prepend to activities (most recent first)
                        this.missionControl.activities.unshift(activity);
                        // Keep only last 50
                        if (this.missionControl.activities.length > 50) {
                            this.missionControl.activities.pop();
                        }
                    }

                    // Refresh icons for activity feed
                    this.$nextTick(() => {
                        if (window.refreshIcons) window.refreshIcons();
                    });
                }
            },

            // ==================== Agent Activity Sheet ====================

            /**
             * Get the first running task (for the banner display)
             */
            getFirstRunningTask() {
                const runningTaskIds = Object.keys(this.missionControl.runningTasks);
                if (runningTaskIds.length === 0) return null;

                const taskId = runningTaskIds[0];
                const runningData = this.missionControl.runningTasks[taskId];
                const task = this.missionControl.tasks.find(t => t.id === taskId);

                return {
                    taskId: taskId,
                    agentName: runningData?.agentName || 'Agent',
                    agentId: runningData?.agentId,
                    taskTitle: task?.title || runningData?.taskTitle || 'Task',
                    startedAt: runningData?.startedAt,
                    outputCount: runningData?.output?.length || 0
                };
            },

            /**
             * Get count of running tasks
             */
            getRunningTaskCount() {
                return Object.keys(this.missionControl.runningTasks).length;
            },

            /**
             * Open the Agent Activity Sheet for a specific task
             */
            openAgentActivitySheet(taskId) {
                const runningData = this.missionControl.runningTasks[taskId];
                const task = this.missionControl.tasks.find(t => t.id === taskId);

                if (!runningData && !task) return;

                this.missionControl.activeAgentTask = {
                    taskId: taskId,
                    agentId: runningData?.agentId,
                    agentName: runningData?.agentName || 'Agent',
                    taskTitle: task?.title || 'Task',
                    startedAt: runningData?.startedAt
                };
                this.missionControl.showAgentActivitySheet = true;

                // Auto-scroll output on open
                this.$nextTick(() => {
                    const panel = this.$refs.agentActivityOutput;
                    if (panel) panel.scrollTop = panel.scrollHeight;
                    if (window.refreshIcons) window.refreshIcons();
                });
            },

            /**
             * Close the Agent Activity Sheet
             */
            closeAgentActivitySheet() {
                this.missionControl.showAgentActivitySheet = false;
                this.missionControl.activeAgentTask = null;
            },

            /**
             * Get full output for the Agent Activity Sheet
             */
            getAgentActivityOutput(taskId) {
                const runningData = this.missionControl.runningTasks[taskId];
                if (!runningData || !runningData.output) return 'Waiting for output...';

                return runningData.output.map(chunk => {
                    if (chunk.type === 'tool_use') {
                        return `🔧 ${chunk.content}`;
                    } else if (chunk.type === 'tool_result') {
                        return `✅ ${chunk.content}`;
                    }
                    return chunk.content;
                }).join('');
            },

            /**
             * Format elapsed time since task started
             */
            formatElapsedTime(startedAt) {
                if (!startedAt) return '';
                const start = new Date(startedAt);
                const now = new Date();
                const diff = Math.floor((now - start) / 1000);

                if (diff < 60) return `${diff}s`;
                if (diff < 3600) return `${Math.floor(diff / 60)}m ${diff % 60}s`;
                return `${Math.floor(diff / 3600)}h ${Math.floor((diff % 3600) / 60)}m`;
            },

            // ==================== Comments/Thread ====================

            /**
             * Load messages for a task
             */
            async loadTaskMessages(taskId) {
                if (!taskId) return;

                this.missionControl.messagesLoading = true;
                this.missionControl.taskMessages = [];

                try {
                    const res = await fetch(`/api/mission-control/tasks/${taskId}/messages`);
                    if (res.ok) {
                        const data = await res.json();
                        this.missionControl.taskMessages = data.messages || [];
                    }
                } catch (e) {
                    console.error('Failed to load messages:', e);
                } finally {
                    this.missionControl.messagesLoading = false;
                    this.$nextTick(() => {
                        const panel = this.$refs.taskMessagesPanel;
                        if (panel) panel.scrollTop = panel.scrollHeight;
                        if (window.refreshIcons) window.refreshIcons();
                    });
                }
            },

            /**
             * Post a message to a task thread
             */
            async postTaskMessage(taskId) {
                const content = this.missionControl.messageInput.trim();
                if (!content || !taskId) return;

                try {
                    // Use 'human' as a special agent ID for human messages
                    const res = await fetch(`/api/mission-control/tasks/${taskId}/messages`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            from_agent_id: 'human',
                            content: content,
                            attachment_ids: []
                        })
                    });

                    if (res.ok) {
                        const data = await res.json();
                        this.missionControl.taskMessages.push(data.message);
                        this.missionControl.messageInput = '';

                        // Scroll to bottom
                        this.$nextTick(() => {
                            const panel = this.$refs.taskMessagesPanel;
                            if (panel) panel.scrollTop = panel.scrollHeight;
                        });
                    } else {
                        const err = await res.json();
                        this.showToast(err.detail || 'Failed to post message', 'error');
                    }
                } catch (e) {
                    console.error('Failed to post message:', e);
                    this.showToast('Failed to post message', 'error');
                }
            },

            // ==================== Deliverables ====================

            /**
             * Load deliverables (documents) for a task
             */
            async loadTaskDeliverables(taskId) {
                if (!taskId) return;

                this.missionControl.deliverablesLoading = true;
                this.missionControl.taskDeliverables = [];

                try {
                    const res = await fetch(`/api/mission-control/tasks/${taskId}/documents`);
                    if (res.ok) {
                        const data = await res.json();
                        this.missionControl.taskDeliverables = data.documents || [];
                    }
                } catch (e) {
                    console.error('Failed to load deliverables:', e);
                } finally {
                    this.missionControl.deliverablesLoading = false;
                    this.$nextTick(() => {
                        if (window.refreshIcons) window.refreshIcons();
                    });
                }
            }
        };
    }
};

window.PocketPaw.Loader.register('MissionControl', window.PocketPaw.MissionControl);
