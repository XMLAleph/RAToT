from flask import Flask, Response
import time

app = Flask(__name__)

@app.route('/stream')
def stream_data():
    def generate():
        # 生成数据的代码
        yield 'First chunk of data '
        time.sleep(5)
        yield 'Second chunk of data '
        time.sleep(5)
        yield 'Third chunk of data '
        time.sleep(5)
        # 这里可以是一个循环，不断产生数据
        yield 'Final chunk of data '

    return Response(generate())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)