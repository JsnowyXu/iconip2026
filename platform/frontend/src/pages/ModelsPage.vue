<template>
  <div class="page-grid two-col">
    <SectionCard title="模型资产库" subtitle="自动发现训练输出的 best_model.pt 与 metrics.json。">
      <div class="button-row">
        <button class="button-primary" @click="syncModels">同步模型</button>
      </div>
      <div class="card-grid compact-grid">
        <button v-for="item in models" :key="item.id" class="dataset-card" @click="selectModel(item)">
          <div class="card-top">
            <span class="tag">{{ item.dataset_key || 'unknown' }}</span>
            <strong>{{ item.name }}</strong>
          </div>
          <p>{{ item.task_key }}</p>
          <div class="mini-stats">
            <span>Inference {{ item.supports_inference ? 'yes' : 'no' }}</span>
            <span>AP {{ item.metrics?.ap ?? '-' }}</span>
          </div>
        </button>
      </div>
    </SectionCard>

    <SectionCard :title="selectedModel?.name || '模型推理'" subtitle="支持链路分数推理与节点标签预测。">
      <div class="form-grid">
        <label>
          模式
          <select v-model="infer.mode">
            <option value="link">链路预测</option>
            <option value="node">节点分类</option>
          </select>
        </label>
        <label v-if="infer.mode === 'link'">
          Source Node
          <input v-model.number="infer.src" type="number" />
        </label>
        <label v-if="infer.mode === 'link'">
          Target Node
          <input v-model.number="infer.dst" type="number" />
        </label>
        <label v-if="infer.mode === 'link'">
          Relation ID
          <input v-model.number="infer.rel" type="number" />
        </label>
        <label v-if="infer.mode === 'node'">
          Node ID
          <input v-model.number="infer.node_id" type="number" />
        </label>
      </div>
      <div class="button-row">
        <button class="button-primary" :disabled="!selectedModel" @click="doInfer">执行推理</button>
      </div>
      <pre class="log-box">{{ JSON.stringify(inferResult, null, 2) }}</pre>
    </SectionCard>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { apiGet, apiPost } from '../api'
import SectionCard from '../components/SectionCard.vue'

const models = ref([])
const selectedModel = ref(null)
const inferResult = ref({})
const infer = ref({ mode: 'link', src: 0, dst: 1, rel: 0, node_id: 0 })

async function loadModels(sync = false) {
  models.value = (await apiGet(`/api/models/${sync ? '?sync=1' : ''}`)).items || []
  if (!selectedModel.value && models.value.length) selectedModel.value = models.value[0]
}

async function syncModels() {
  await apiPost('/api/models/discover/', {})
  await loadModels()
}

function selectModel(item) {
  selectedModel.value = item
}

async function doInfer() {
  if (!selectedModel.value) return
  inferResult.value = await apiPost(`/api/models/${selectedModel.value.id}/infer/`, infer.value)
}

onMounted(async () => {
  await loadModels(true)
})
</script>
