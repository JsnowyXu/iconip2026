<template>
  <div class="job-table-wrap">
    <table class="job-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>名称</th>
          <th>类型</th>
          <th>状态</th>
          <th>输出</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="job in jobs" :key="job.id" @click="$emit('select', job)">
          <td>{{ job.id }}</td>
          <td>{{ job.name }}</td>
          <td>{{ job.kind }}</td>
          <td><span :class="`status status-${job.status}`">{{ job.status }}</span></td>
          <td class="truncate">{{ job.output_path }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup>
defineProps({ jobs: { type: Array, default: () => [] } })
defineEmits(['select'])
</script>
