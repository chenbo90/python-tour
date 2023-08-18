#如果没有安装，请执行pip install flask
from flask import Flask, request
app = Flask(__name__)
@app.route('/hello', methods=['GET', 'POST'])
def hello():
    print(request.method)
    if request.method == 'GET':
        name = request.args.get('name')
        print('GET方式接收参数，name值为：'+name)
        return {'message': f'Hello, {name}!'}
    elif request.method== 'POST':
        name = request.json.get("name")
        print('POST方式接收参数，name值为：'+name)
        return {'message': f'Hello, {name}!'}


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=9080)