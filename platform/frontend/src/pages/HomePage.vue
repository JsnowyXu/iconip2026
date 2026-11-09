<template>
  <div class="page-grid page-home">
    <SectionCard class="hero-card">
      <div class="hero-layout">
        <div>
          <div class="eyebrow">答辩展示系统</div>
          <h1>异质动态图社区学习的一站式实验产品化平台</h1>
          <p class="hero-text">
            从原始数据集展示、预处理管道、模型训练到模型推理，全部围绕当前 DHC 算法体系构建，适合实验管理、结果演示和毕业答辩展示。
          </p>
          <a class="primary-link" :href="home.github_url" target="_blank">查看 GitHub</a>
        </div>
        <StatsStrip :items="heroStats" />
      </div>
    </SectionCard>

    <SectionCard title="平台亮点" subtitle="结合当前算法代码直接落地为工程产品页面。">
      <div class="feature-grid">
        <div v-for="item in home.highlights || []" :key="item" class="feature-box">{{ item }}</div>
      </div>
    </SectionCard>

    <SectionCard title="README 摘要" subtitle="首页直接呈现必要说明，便于演示时快速切入。">
      <pre class="readme-box">{{ home.readme_excerpt }}</pre>
    </SectionCard>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { apiGet } from '../api'
import SectionCard from '../components/SectionCard.vue'
import StatsStrip from '../components/StatsStrip.vue'

const home = ref({ highlights: [] })
const heroStats = computed(() => [
  { label: '数据集', value: home.value.dataset_count ?? '-' },
  { label: '任务记录', value: home.value.job_count ?? '-' },
  { label: '模型资产', value: home.value.model_count ?? '-' },
])

onMounted(async () => {
  home.value = await apiGet('/api/home/')
})
</script>
