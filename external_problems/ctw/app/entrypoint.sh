/root/.local/bin/ray start --head --dashboard-host="0.0.0.0" --num-cpus 4
RAY_task_oom_retries=5 /root/.local/bin/serve run -h 0.0.0.0 -p 9527 translate_service:translator
