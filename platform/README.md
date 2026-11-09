# DHC Platform

前后端分离的算法展示与调度系统，围绕当前 `dyn_hetero_community` 代码实现构建。

## 技术栈

- 前端：Vue 3 + Vite
- 后端：Django
- 数据库：MySQL 8
- 调度：Django API + Background Worker
- 部署：Docker Compose

## 页面能力

- 首页：项目摘要、GitHub 链接、README 摘要
- 数据集页：数据卡片、统计信息、任务适配、图结构预览
- 预处理页：自动生成 `prepare_dataset.py` / task builder 脚本并执行
- 训练页：自动生成 `python -m dhc.train.run_train` 脚本并入队
- 推理页：发现模型资产，执行链路预测或节点分类推理

## 一键部署

```bash
cd /home/xuxj/2026/dyn_hetero_community/platform
docker compose up --build
```

启动后访问：

- Frontend: `http://localhost:8080`
- Backend API: `http://localhost:8000/api/health/`
- MySQL: `localhost:3307`

## 本地开发

后端：

```bash
cd /home/xuxj/2026/dyn_hetero_community/platform/backend
pip install -r ../../requirements.txt -r requirements.txt
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

Worker：

```bash
cd /home/xuxj/2026/dyn_hetero_community/platform/backend
python manage.py runjobworker --poll-sec 3
```

前端：

```bash
cd /home/xuxj/2026/dyn_hetero_community/platform/frontend
npm install
npm run dev
```

## 说明

- 训练和推理会直接复用仓库内的 `src/dhc`、`scripts/datasets` 和 `scripts/experiments`。
- 预处理输出默认写入 `data/processed/platform/` 与 `data/processed/platform_tasks/`。
- 训练输出默认写入 `results/platform_runs/`。
