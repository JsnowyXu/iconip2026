<template>
  <div class="page-grid two-col">
    <SectionCard title="预处理管道" subtitle="选择数据集、任务模式和文本编码，平台自动生成并执行脚本。">
      <div class="form-grid">
        <label>
          数据集
          <select v-model="form.dataset_key">
            <option v-for="item in options.datasets || []" :key="item.key" :value="item.key">{{ item.label }}</option>
          </select>
        </label>
        <label>
          管道模式
          <select v-model="form.mode">
            <option v-for="mode in options.pipeline_modes || []" :key="mode.key" :value="mode.key">{{ mode.label }}</option>
          </select>
        </label>
        <label>
          原始归档路径
          <input v-model="form.archive_path" placeholder="/home/.../dataset.zip" />
        </label>
        <label>
          统一图输入路径
          <input v-model="form.input_unified_path" placeholder="仅 build_tasks 时必填" />
        </label>
        <label>
          最大边数
          <input v-model.number="form.max_edges" type="number" />
        </label>
        <label>
          文本编码器
          <select v-model="form.text_encoder">
            <option v-for="item in options.text_encoders || []" :key="item.key" :value="item.key">{{ item.label }}</option>
          </select>
        </label>
        <label>
          文本维度
          <input v-model.number="form.text_dim" type="number" />
        </label>
        <label>
          句向量模型路径
          <input v-model="form.text_model_name" placeholder="可选" />
        </label>
      </div>
      <div class="button-row">
        <button class="button-primary" @click="submitJob">提交预处理任务</button>
      </div>
      <pre class="command-box">{{ selectedJob?.command_preview || '提交后将在这里显示自动生成脚本。' }}</pre>
    </SectionCard>

    <SectionCard title="任务队列" subtitle="Worker 将自动执行并写入日志。">
      <JobTable :jobs="jobs" @select="selectJob" />
      <pre class="log-box">{{ jobLog }}</pre>
    </SectionCard>

    <SectionCard class="full-span" title="输出图预览" subtitle="预处理成功后可以直接查看输出结构。">
      <GraphCanvas :graph="selectedJob?.result?.graph_preview || { nodes: [], edges: [] }" />
    </SectionCard>
  </div>
</template>

<script setup>
import { onMounted, ref, watch } from 'vue'
import { apiGet, apiPost } from '../api'
import GraphCanvas from '../components/GraphCanvas.vue'
import JobTable from '../components/JobTable.vue'
import SectionCard from '../components/SectionCard.vue'

const options = ref({ datasets: [], pipeline_modes: [], text_encoders: [] })
const jobs = ref([])
const selectedJob = ref(null)
const jobLog = ref('')
const form = ref({
  dataset_key: 'raw_cn',
  mode: 'build_tasks',
  archive_path: '',
  input_unified_path: '',
  max_edges: 120000,
  text_encoder: 'hybrid_hash',
  text_dim: 64,
  text_model_name: '',
})

function getDataset(key) {
  return (options.value.datasets || []).find((item) => item.key === key) || null
}

function syncDatasetDefaults(force = false) {
  const dataset = getDataset(form.value.dataset_key)
  if (!dataset) return
  if ((force || !form.value.input_unified_path) && dataset.unified_path) {
    form.value.input_unified_path = dataset.unified_path
  }
  if (form.value.mode === 'build_tasks') {
    form.value.archive_path = ''
  }
}

async function refreshJobs() {
  jobs.value = (await apiGet('/api/jobs/?kind=preprocess')).items || []
}

async function selectJob(job) {
  selectedJob.value = await apiGet(`/api/jobs/${job.id}/`)
  const log = await apiGet(`/api/jobs/${job.id}/log/`)
  jobLog.value = log.log || ''
}

async function submitJob() {
  const res = await apiPost('/api/pipelines/preprocess/', form.value)
  selectedJob.value = res.job
  await refreshJobs()
}

watch(() => form.value.dataset_key, () => {
  syncDatasetDefaults(true)
})

watch(() => form.value.mode, (mode) => {
  if (mode === 'build_tasks') {
    syncDatasetDefaults(false)
  }
})

onMounted(async () => {
  options.value = await apiGet('/api/pipelines/options/')
  const datasetKeys = (options.value.datasets || []).map((item) => item.key)
  if (!datasetKeys.includes(form.value.dataset_key) && datasetKeys.length) {
    form.value.dataset_key = datasetKeys[0]
  }
  syncDatasetDefaults(true)
  await refreshJobs()
})
</script>
