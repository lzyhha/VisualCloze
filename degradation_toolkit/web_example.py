import json
import os
import random
import sys
from pathlib import Path

import pandas
from data_reader import read_img_general
from flask import Flask, make_response, request

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index_page():
    ps ='''
<a href="/leo/0">leonardo</a> <br />
'''
    return f"""<body>
    Subsets
  <p>
  {ps}
</p>
</body>"""


@app.route("/<name>/<int:page>/")
def show_page(name,page):
    table_rows = ""

    urls = []
    captions = []

    if name=='leo':
        with open('omnigen/data_json/meta_info_imageEdit_ultraEdit_4M_part2.json')as f:
            data = json.load(f)
        for d in data:
            # url = '/data3/huxiangfei/sample/'+ d['video_path'].split('/')[-1]
            url = d['input_path']
            urls.append(url)
            # url = d['target_path']
            # urls.append(url)
            c = ''
            for k in list(d.keys()):
                c += f'{k}: {d[k]}\n\n'
            captions.append(c)



    st, ed = page * 20, (page + 1) * 20
    for url0, caption in zip(urls[st:ed], captions[st: ed]):
        images = f"""<img src="/imgs/{url0}" style="width: 800px">"""

        table_rows += f"""<tr>
    <td style="text-align: center;vertical-align: middle;">{images}</td>
        <td><pre style="white-space: pre-wrap;width: 800px;">{caption}</pre></td>
</tr>"""
    return f"""<head>
    <style>
    table, th, td {{
        border: 1px solid black;
        border-collapse: collapse;
        padding-left: 10px;
        padding-right: 10px;
    }};
    </style>
</head>
<body>
    <p><a href="/">home</a> |
    <a href="/{name}/{max(page - 1, 0)}">previous</a> | 
    <a href="/{name}/{page + 1}">next</a>
    <table>
    <tr>
        <th style="width: 400px">Image</th>
        <th>Annotation</th>
    </tr>
    {table_rows}
    </table>
    <a href="/">home</a> |
    <a href="/{name}/{max(page - 1, 0)}">previous</a> | 
    <a href="/{name}/{page + 1}">next</a>
</body>
"""


# @app.route("/imgs/<path:path>")
# def show_img(path):
#     #return open('/'+path, "rb").read()
#     print(f"image_path:{path}")
#     return read_img_general(path)
#     # if os.path.exists('/public/home/qiult/projects/VLDGen/'):
#     #     return open('/public/home/qiult/projects/VLDGen/'+path, "rb").read()
#     # else:
#     #     return open(r'D:\Coding\HOIBaseline\VLDGen\\' + path, "rb").read()

@app.route("/imgs/<path:path>")
def show_img(path):
    try:
        # 调用 read_img_general 函数
        img_data = read_img_general(path)
        
        # 检查是否读取成功
        if img_data:
            response = make_response(img_data)
            response.headers.set('Content-Type', 'image/jpeg')  # 假设图片是 JPEG 格式
            return response
        else:
            return "Image not found", 404
    except Exception as e:
        # 打印错误信息并返回错误响应
        print(f"Error processing image {path}: {e}")
        return f"Error processing image: {str(e)}", 500


# @app.route("/imgs/<path:path>")
# def show_img(path):
#     return open('/'+path, "rb").read()

app.run(
    host="127.0.0.1",
    port=7848,
    debug=True,
)