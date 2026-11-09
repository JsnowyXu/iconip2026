import { createRouter, createWebHistory } from 'vue-router'
import HomePage from '../pages/HomePage.vue'
import DatasetsPage from '../pages/DatasetsPage.vue'
import PipelinesPage from '../pages/PipelinesPage.vue'
import TrainingPage from '../pages/TrainingPage.vue'
import ModelsPage from '../pages/ModelsPage.vue'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: HomePage },
    { path: '/datasets', component: DatasetsPage },
    { path: '/pipelines', component: PipelinesPage },
    { path: '/training', component: TrainingPage },
    { path: '/models', component: ModelsPage },
  ],
})
