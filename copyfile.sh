#!/bin/bash
scp $1 ubuntu@10.21.10.205:/home/ubuntu/chenbo/python_code

[ $? -eq 0 ] || (echo "上传失败,退出"; exit)