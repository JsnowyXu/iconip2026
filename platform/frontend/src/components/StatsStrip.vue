<template>
  <div class="stats-strip">
    <div v-for="item in items" :key="item.label" class="stat-chip">
      <span>{{ item.label }}</span>
      <strong>{{ item.value }}</strong>
    </div>
  </div>
</template>

<script setup>
defineProps({ items: { type: Array, default: () => [] } })
</script>
