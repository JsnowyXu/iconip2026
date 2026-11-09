<template>
  <div class="page-grid two-col">
    <SectionCard title="数据集目录" subtitle="参考 Hugging Face Dataset Card 的卡片式展示方式。">
      <div class="card-grid">
        <button v-for="item in datasets" :key="item.key" class="dataset-card" @click="selectDataset(item.key)">
          <div class="card-top">
            <span class="tag">{{ item.category }}</span>
            <strong>{{ item.label }}</strong>
          </div>
          <p>{{ item.description }}</p>
          <div class="mini-stats">
            <span>Nodes {{ item.stats?.num_nodes ?? '-' }}</span>
            <span>Edges {{ item.stats?.num_edges ?? '-' }}</span>
            <span>Tasks {{ item.task_file_count }}</span>
          </div>
        </button>
      </div>
    </SectionCard>

    <SectionCard :title="detail?.label || '数据集详情'" subtitle="节点、边、标签规模与适配任务。">
      <StatsStrip :items="detailStats" />
      <div v-if="detail" class="detail-block">
        <p>{{ detail.description }}</p>
        <div class="pill-row">
          <span v-for="task in detail.tasks" :key="task.key" class="task-pill">{{ task.label }}</span>
        </div>
        <div class="subgrid">
          <div>
            <h3>标签分布</h3>
            <ul class="plain-list">
              <li v-for="row in detail.distributions?.class_distribution || []" :key="JSON.stringify(row)">
                {{ row.label || row.class || row.category || Object.values(row)[0] }} /
                {{ row.count || Object.values(row)[1] }}
              </li>
            </ul>
          </div>
          <div>
            <h3>任务文件</h3>
            <ul class="plain-list">
              <li v-for="file in detail.task_files || []" :key="file.path">{{ file.name }} · {{ file.size }}</li>
            </ul>
          </div>
        </div>
      </div>
    </SectionCard>

    <SectionCard class="full-span" title="局部图结构预览" subtitle="支持异构关系采样展示，便于答辩解释网络结构。">
      <GraphCanvas :graph="graph" />
    </SectionCard>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { apiGet } from '../api'
import GraphCanvas from '../components/GraphCanvas.vue'
import SectionCard from '../components/SectionCard.vue'
import StatsStrip from '../components/StatsStrip.vue'

const datasets = ref([])
const detail = ref(null)
const graph = ref({ nodes: [], edges: [] })

const detailStats = computed(() => {
  const stats = detail.value?.stats || {}
  return [
    { label: '节点数', value: stats.num_nodes ?? '-' },
    { label: '边数', value: stats.num_edges ?? '-' },
    { label: '关系数', value: stats.num_relations ?? '-' },
    { label: '特征维度', value: stats.node_feat_dim ?? '-' },
    { label: '统一文件', value: detail.value?.unified_size ?? '-' },
  ]
})

async function selectDataset(key) {
  detail.value = await apiGet(`/api/datasets/${key}/`)
  graph.value = await apiGet(`/api/datasets/${key}/graph/`)
}

onMounted(async () => {
  const data = await apiGet('/api/datasets/')
  datasets.value = data.items || []
  if (datasets.value.length) await selectDataset(datasets.value[0].key)
})
</script>
