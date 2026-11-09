<template>
  <div class="page-grid two-col">
    <SectionCard title="模型算法构建" subtitle="选择任务数据与算法配置，平台自动生成 run_train 脚本并加入队列。">
      <div class="form-grid">
        <label>
          任务数据
          <select v-model="form.data_path">
            <option v-for="item in options.task_candidates || []" :key="item.data_path" :value="item.data_path">
              {{ item.dataset_label }} / {{ item.task_name }}
            </option>
          </select>
        </label>
        <label>
          运行名称
          <input v-model="form.run_name" placeholder="可选" />
        </label>
        <label>
          Epochs
          <input v-model.number="form.epochs" type="number" />
        </label>
        <label>
          Batch Size
          <input v-model.number="form.batch_size" type="number" />
        </label>
        <label>
          Learning Rate
          <input v-model.number="form.lr" type="number" step="0.0001" />
        </label>
        <label>
          GPU ID
          <input v-model.number="form.gpu_id" type="number" />
        </label>
        <label>
          Recent Neighbor Size
          <input v-model.number="form.recent_neighbor_size" type="number" />
        </label>
        <label>
          Lambda CL
          <input v-model.number="form.lambda_cl" type="number" step="0.01" />
        </label>
        <label>
          Lambda ML
          <input v-model.number="form.lambda_ml" type="number" step="0.1" />
        </label>
        <label class="checkbox-row">
          <input v-model="form.ml_threshold_search" type="checkbox" /> 多标签阈值搜索
        </label>
      </div>
      <div class="button-row">
        <button class="button-primary" @click="submitTraining">启动训练</button>
      </div>
      <pre class="command-box">{{ selectedJob?.command_preview || '训练脚本会在这里自动生成。' }}</pre>
    </SectionCard>

    <SectionCard title="训练任务列表" subtitle="查看日志、指标和输出目录。">
      <JobTable :jobs="jobs" @select="selectJob" />
      <pre class="log-box">{{ jobLog }}</pre>
    </SectionCard>

    <SectionCard class="full-span" title="结果摘要" subtitle="训练成功后会自动归档模型资产与指标。">
      <div class="result-grid" v-if="selectedJob?.result">
        <div class="result-box"><span>输出目录</span><strong>{{ selectedJob.output_path }}</strong></div>
        <div class="result-box"><span>Artifact ID</span><strong>{{ selectedJob.result.artifact_id || '-' }}</strong></div>
        <div class="result-box"><span>Artifact Name</span><strong>{{ selectedJob.result.artifact_name || '-' }}</strong></div>
      </div>
    </SectionCard>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { apiGet, apiPost } from '../api'
import JobTable from '../components/JobTable.vue'
import SectionCard from '../components/SectionCard.vue'

const options = ref({ task_candidates: [], defaults: {} })
const jobs = ref([])
const selectedJob = ref(null)
const jobLog = ref('')
const form = ref({
  data_path: '',
  run_name: '',
  epochs: 5,
  batch_size: 512,
  lr: 0.001,
  gpu_id: 0,
  recent_neighbor_size: 12,
  lambda_cl: 0.05,
  lambda_ml: 0.7,
  community_assign_temperature: 1.0,
  community_assign_topk: 0,
  typed_negative_ratio: 0.6,
  popular_negative_ratio: 0.2,
  ml_threshold_search: true,
  ml_pos_weight: true,
})

async function refreshJobs() {
  jobs.value = (await apiGet('/api/jobs/?kind=training')).items || []
}

async function selectJob(job) {
  selectedJob.value = await apiGet(`/api/jobs/${job.id}/`)
  const log = await apiGet(`/api/jobs/${job.id}/log/`)
  jobLog.value = log.log || ''
}

async function submitTraining() {
  const res = await apiPost('/api/training/jobs/', form.value)
  selectedJob.value = res.job
  await refreshJobs()
}

onMounted(async () => {
  options.value = await apiGet('/api/training/options/')
  Object.assign(form.value, options.value.defaults || {})
  if ((options.value.task_candidates || []).length) {
    form.value.data_path = options.value.task_candidates[0].data_path
  }
  await refreshJobs()
})
</script>
