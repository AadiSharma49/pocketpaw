/**
 * PocketPaw - Skills Feature Module
 *
 * Created: 2026-02-05
 * Extracted from app.js as part of componentization refactor.
 *
 * Contains skill-related state and methods:
 * - Skill listing and management
 * - Skill execution
 * - Skill command parsing
 */

window.PocketPaw = window.PocketPaw || {};

window.PocketPaw.Skills = {
    name: 'Skills',
    /**
     * Get initial state for Skills
     */
    getState() {
        return {
            showSkills: false,
            skills: [],
            skillsLoading: false
        };
    },

    /**
     * Get methods for Skills
     */
    getMethods() {
        return {
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
            }
        };
    }
};

window.PocketPaw.Loader.register('Skills', window.PocketPaw.Skills);
