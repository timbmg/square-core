/**
 * Contains all API calls
 * Endpoint documentation is found in the documentation of the backend and skill server
 */
import axios from 'axios'

/**
 * URLs to the SQuARE backend servers
 */
const SKILL_URL = `${process.env.VUE_APP_URL}/api/skill-manager`

/**
 * Get a list of available skill types.
 * @param {Object} headers optional authentication header
 */
export function getSkillTypes(headers) {
    return axios.get(`${SKILL_URL}/skill-types`, { headers: headers })
}

/**
 * Get a list of available skills. 
 * The user name is only required for unpublished skills of the user. Published skills are available without.
 * @param {Object} headers optional authentication header
 */
export function getSkills(headers) {
    return axios.get(`${SKILL_URL}/skill`, { headers: headers })
}

/**
 * Get a single skill.
 * @param {Object} headers optional authentication header
 * @param {String} skillId ID of the skill
 */
export function getSkill(headers, skillId) {
    return axios.get(`${SKILL_URL}/skill/${skillId}`, { headers: headers })
}

/**
 * Permanently deletes the skill with the given ID.
 * @param {Object} headers optional authentication header
 * @param {String} skillId ID of the skill that will be deleted
 */
export function deleteSkill(headers, skillId) {
    return axios.delete(`${SKILL_URL}/skill/${skillId}`, { headers: headers })
}

/**
 * Updates the skill with the given ID with the new values. 
 * Only skills with owner ID as specified in access token can be updated.
 * @param {Object} headers optional authentication header
 * @param {String} skillId ID of the skill that will be updated
 * @param {Object} newSkill the new values for the skill. All fields need to be present. If a value should not be updated, then set the old value there.
 */
export function putSkill(headers, skillId, newSkill) {
    return axios.put(`${SKILL_URL}/skill/${skillId}`, newSkill, { headers: headers })
}

/**
 * Sends a question to the backend and receives the resulting answers
 * @param {Object} headers optional authentication header
 * @param {String} question the asked question
 * @param {String} context the provided context
 * @param {Object} options the options for the request
 */
export function postQuery(headers, question, context, options) {
    let data = {
        query: question,
        skill_args: {},
        num_results: options.maxResultsPerSkill
    }
    if (context.length > 0) {
        data.skill_args.context = context
    }
    let results = options.selectedSkills.map(skillId => {
        return axios.post(`${SKILL_URL}/skill/${skillId}/query`, data, { headers: headers })
    })
    return axios.all(results)
}

/**
 * Creates a new skill for the owner as specified in the authentication header.
 * @param {Object} headers optional authentication header
 * @param {Object} newSkill the values for the new skill
 */
export function postSkill(headers, newSkill) {
    return axios.post(`${SKILL_URL}/skill`, newSkill, { headers: headers })
}

/**
 * Ping skill server to check for availability.
 * @param {Object} headers optional authentication header
 * @param {String} skillUrl URL to the skill server. Format: {scheme}://host[:port]/{base_path}
 */
export function pingSkill(headers, skillUrl) {
    headers.params = { skill_url: skillUrl }
    return axios.get(`${SKILL_URL}/health/skill-heartbeat`, headers)
}
