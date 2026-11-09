<template>
  <div class="app-shell">
    <header class="topbar">
      <div>
        <div class="eyebrow">DHC Product Platform</div>
        <div class="brand">Dynamic Heterogeneous Community Studio</div>
      </div>
      <nav class="nav">
        <RouterLink to="/">首页</RouterLink>
        <RouterLink to="/datasets">数据集</RouterLink>
        <RouterLink to="/pipelines">预处理</RouterLink>
        <RouterLink to="/training">训练</RouterLink>
        <RouterLink to="/models">推理</RouterLink>
      </nav>
    </header>
    <main class="page-wrap">
      <slot />
    </main>
  </div>
</template>
