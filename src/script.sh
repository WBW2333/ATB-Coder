#!/bin/bash

# 无限循环，直到你手动停止脚本
while true
do
    # 运行 Python 程序
    python run_benchmark.py \
        --problem ../LeetCode-Benchmark/Total.json \
        --ATP ../LeetCode-Benchmark/Template.json \
        --tag Total \
        --info Test

    # 获取上一个命令的退出状态
    exit_code=$?

    # 检查 Python 程序的退出状态
    if [ $exit_code -ne 0 ]; then
        echo "程序意外中断，正在重新启动..."
    else
        # 正常退出
        echo "程序正常完成。"
        break
    fi

    # 可以添加一个延迟，避免频繁重启
    sleep 1
done
