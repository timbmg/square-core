<template>
  <div>
    <form v-on:submit.prevent="showCheckList">
      <div class="row">
        <div class="col">
          <CompareSkills
              selector-target="explain"
              :skill-filter="skillId => skillId in checklistData"
              v-on:input="changeSelectedSkills"
              class="border-success" />
        </div>
      </div>
      <div v-if="selectedSkills.length > 0" class="row">
        <div class="col mt-4">
          <div class="d-grid gap-2 d-md-flex justify-content-md-center">
            <button type="submit" class="btn btn-success btn-lg shadow text-white" :disabled="waiting">
              <span v-show="waiting" class="spinner-border spinner-border-sm" role="status" />
              &nbsp;Show CheckList</button>
          </div>
        </div>
      </div>
    </form>
    <div v-if="currentTests.length > 0">
      <div class="row">
        <div class="col table-responsive bg-light border border-primary rounded shadow p-3 mx-3 mt-4">
          <table class="table table-borderless">
            <thead class="border-bottom border-dark">
            <tr>
              <th
                  v-for="(skill, index) in currentSkills"
                  :key="index"
                  scope="col"
                  class="fs-2 fw-light text-center">{{ skill.name }}</th>
            </tr>
            <tr>
              <th
                  v-for="index in currentSkills.length"
                  :key="index"
                  scope="col"
                  class="fw-normal text-center">
                <a
                    v-on:click="downloadExamples(index)"
                    :ref="`downloadButton${index}`"
                    class="btn btn-outline-secondary d-inline-flex align-items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="currentColor" class="bi bi-download" viewBox="0 0 16 16">
                    <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                    <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
                  </svg>
                  &nbsp;Download all examples
                </a>
              </th>
            </tr>
            <tr>
              <th
                  v-for="(skill, index) in currentSkills"
                  :key="index"
                  scope="col"
                  class="fw-normal text-center">{{ skill.description }}</th>
            </tr>
            </thead>
            <tbody>
            <tr
                v-for="row in currentTests[0].length"
                :key="row">
              <td
                  v-for="index in currentSkills.length"
                  :key="index"
                  :width="`${100 / currentSkills.length }%`"
                  style="min-width: 320px;">
                <div class="progress flex-grow-1 align-self-center m-2" title="Failure rate">
                  <div
                      class="progress-bar bg-danger"
                      role="progressbar"
                      :style="{ width: `${roundScore(getTest(index, row).failed_cases / getTest(index, row).total_cases)}%` }"
                      :aria-valuenow="roundScore(getTest(index, row).failed_cases / getTest(index, row).total_cases)"
                      aria-valuemin="0"
                      aria-valuemax="100">{{ getTest(index, row).failed_cases }}</div>
                  <div
                      class="progress-bar bg-success"
                      role="progressbar"
                      :style="{ width: `${roundScore(getTest(index, row).success_cases / getTest(index, row).total_cases)}%` }"
                      :aria-valuenow="roundScore(getTest(index, row).success_cases / getTest(index, row).total_cases)"
                      aria-valuemin="0"
                      aria-valuemax="100">{{ getTest(index, row).success_cases }}</div>
                </div>
                <div class="text-center">
                  <h3 class="my-3">{{ getTest(index, row).test_name }}</h3>
                  <p class="d-inline-flex align-items-center">
                    <BadgePopover :popover-title="mapTestType(getTest(index, row).test_type)" :popover-content="getTest(index, row).test_type_description" />
                    test on
                    <BadgePopover :popover-title="getTest(index, row).capability" :popover-content="getTest(index, row).capability_description" />
                  </p>
                  <div>
                  <a
                      class="btn btn-outline-secondary d-inline-flex align-items-center"
                      data-bs-toggle="modal"
                      :data-bs-target="`#modal-${index}-${row}`"
                      role="button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="currentColor" class="bi bi-arrows-angle-expand" viewBox="0 0 16 16">
                      <path fill-rule="evenodd" d="M5.828 10.172a.5.5 0 0 0-.707 0l-4.096 4.096V11.5a.5.5 0 0 0-1 0v3.975a.5.5 0 0 0 .5.5H4.5a.5.5 0 0 0 0-1H1.732l4.096-4.096a.5.5 0 0 0 0-.707zm4.344-4.344a.5.5 0 0 0 .707 0l4.096-4.096V4.5a.5.5 0 1 0 1 0V.525a.5.5 0 0 0-.5-.5H11.5a.5.5 0 0 0 0 1h2.768l-4.096 4.096a.5.5 0 0 0 0 .707z"/>
                    </svg>
                    &nbsp;Expand
                  </a>
                    </div>
                </div>
                <ExplainDetail :id="`modal-${index}-${row}`" :test="getTest(index, row)" />
              </td>
            </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
    <div v-else class="row">
      <div class="col-md-8 mx-auto mt-4 text-center">
        <div class="bg-light border rounded shadow p-5 text-center">
          <div class="feature-icon bg-success bg-gradient">
            <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="currentColor" class="bi bi-search" viewBox="0 0 16 16">
              <path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z"/>
            </svg>
          </div>
          <h2 class="display-5">Explainability</h2>
          <p class="lead fs-2">For now we are testing the <span class="text-success">behaviour</span> of <span class="text-success">black-box</span> models with more to come.</p>
          <p class="lead fs-2">Explore capabilities such as the <span class="text-success">robustness</span> of model output.</p>
          <p class="lead fs-2"><span class="text-success">Get started</span> by selecting up to three skills.</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import Vue from 'vue'
import BadgePopover from '../components/BadgePopover'
import CompareSkills from '../components/CompareSkills'
import ExplainDetail from '../components/modals/ExplainDetail'
import mixin from '../components/results/mixin'
import { getSkill } from '../api'

export default Vue.component('explainability-page', {
  mixins: [mixin],
  data() {
    return {
      waiting: false,
      options: {
        selectedSkills: []
      },
      currentSkills: [],
      currentTests: [],
      selectedTest: -1
    }
  },
  components: {
    ExplainDetail,
    BadgePopover,
    CompareSkills
  },
  computed: {
    availableSkills() {
      return this.$store.state.availableSkills
    },
    selectedSkills() {
      return this.options.selectedSkills.filter(skill => skill !== 'None')
    },
    checklistData() {
      // Dynamically require available CheckList data
      let requireComponent = require.context('../../checklist', false, /[a-z0-9]+\.json$/)
      return Object.assign({}, ...requireComponent.keys().map(
          fileName => ({[fileName.substr(2, fileName.length - 7)]: requireComponent(fileName).tests})))
    }
  },
  methods: {
    changeSelectedSkills(options, skillSettings) {
      skillSettings
      this.options = options
    },
    showCheckList() {
      this.waiting = true
      let currentSkills = []
      let currentTests = []
      this.selectedSkills.forEach(skillId => {
        if (skillId in this.checklistData) {
          getSkill(this.$store.getters.authenticationHeader(), skillId)
              .then((response) => {
                currentSkills.push(response.data)
              })
          let tests = this.checklistData[skillId]
          // Sort first skill by failure rate and subsequent skills based on the sorting of the first one
          if (currentTests.length === 0) {
            tests.sort((a, b) => b.failure_rate - a.failure_rate)
          } else {
            tests.sort((a, b) => currentTests[0].findIndex(e => e.test_name === a.test_name) - currentTests[0].findIndex(e => e.test_name === b.test_name))
          }
          tests.forEach(test => test.test_cases = test.test_cases.filter(
              test_case => test_case['success_failed'] === 'failed'))
          currentTests.push(tests)
        }
      })
      this.currentSkills = currentSkills
      this.currentTests = currentTests
      this.waiting = false
    },
    getTest(skillIndex, testIndex) {
      return this.currentTests[skillIndex - 1][testIndex - 1]
    },
    downloadExamples(skillIndex) {
      let skill = this.currentSkills[skillIndex - 1]
      let data = JSON.stringify(this.checklistData[skill.id], null, 2)
      let blob = new Blob([data], {type: 'application/json;charset=utf-8'})
      this.$refs[`downloadButton${skillIndex}`][0].href = URL.createObjectURL(blob)
      this.$refs[`downloadButton${skillIndex}`][0].download = `${skill.name} ${new Date().toLocaleString().replaceAll(/[\\/:]/g, '-')}.json`
    }
  }
})
</script>