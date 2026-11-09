<template>
  <div class="graph-panel">
    <svg v-if="layout.nodes.length" :viewBox="`0 0 ${width} ${height}`" class="graph-svg">
      <line
        v-for="edge in layout.edges"
        :key="edge.id"
        :x1="edge.x1"
        :y1="edge.y1"
        :x2="edge.x2"
        :y2="edge.y2"
        class="graph-edge"
      />
      <circle
        v-for="node in layout.nodes"
        :key="node.id"
        :cx="node.x"
        :cy="node.y"
        :r="node.r"
        class="graph-node"
      />
      <text
        v-for="node in layout.nodes.slice(0, 16)"
        :key="`label-${node.id}`"
        :x="node.x"
        :y="node.y - 10"
        text-anchor="middle"
        class="graph-label"
      >
        {{ node.id }}
      </text>
    </svg>
    <div v-else class="empty-box">暂无图结构预览</div>
    <div class="legend">节点按采样顺序环形布局，适合答辩展示局部结构与关系密度。</div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  graph: { type: Object, default: () => ({ nodes: [], edges: [] }) },
  width: { type: Number, default: 860 },
  height: { type: Number, default: 360 },
})

const layout = computed(() => {
  const cx = props.width / 2
  const cy = props.height / 2
  const radius = Math.min(props.width, props.height) * 0.36
  const nodes = (props.graph?.nodes || []).map((node, idx, arr) => {
    const angle = (Math.PI * 2 * idx) / Math.max(arr.length, 1)
    return {
      ...node,
      x: cx + Math.cos(angle) * radius,
      y: cy + Math.sin(angle) * radius,
      r: 6 + Math.min(node.degree || 0, 10) * 0.6,
    }
  })
  const nodeMap = Object.fromEntries(nodes.map((node) => [node.id, node]))
  const edges = (props.graph?.edges || [])
    .filter((edge) => nodeMap[edge.source] && nodeMap[edge.target])
    .map((edge) => ({
      ...edge,
      x1: nodeMap[edge.source].x,
      y1: nodeMap[edge.source].y,
      x2: nodeMap[edge.target].x,
      y2: nodeMap[edge.target].y,
    }))
  return { nodes, edges }
})
</script>
