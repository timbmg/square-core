/**
 * Code entry point. Initialize Vue and any other JS components that need it.
 */
import Vue from 'vue'
import App from './App.vue'
import * as bootstrap from 'bootstrap'
import router from './router'
import store from './store'
import Keycloak from 'keycloak-js'

Vue.use(bootstrap)

let initOptions = {
  url: `${process.env.VUE_APP_URL}/auth`,
  realm: 'square',
  clientId: 'web-app',
  onLoad: 'check-sso',
  silentCheckSsoRedirectUri: window.location.origin + '/silent-check-sso.html',
  tokenStore: 'cookie'
}

let keycloak = Keycloak(initOptions)

keycloak.init({
  onLoad: initOptions.onLoad,
  silentCheckSsoRedirectUri: initOptions.silentCheckSsoRedirectUri
}).then((authenticated) => {
  if (authenticated) {
    keycloak.loadUserInfo().then(userInfo => {
      store.dispatch('signIn', { userInfo: userInfo, token: keycloak.token })
    })
    setInterval(() => {
      keycloak.updateToken(70).then(() => {
        store.dispatch('refreshToken', { token: keycloak.token })
      })
    }, 6000)
  }
})

// Init Vue
new Vue({
  router,
  store,
  render: h => h(App, { props: { keycloak: keycloak } })
}).$mount('#app')
