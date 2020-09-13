import warnings
warnings.filterwarnings(action='ignore')

import json
from flask import Flask, make_response
from flask_restful import reqparse, Api, Resource
from service.module import DialogKoBERT,DialogElectra
#flask server
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)


#koBERT 모델 로딩
dialog_kobert = DialogKoBERT()
dialog_electra = DialogElectra()


class DialogKoBERTAPI(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('s')
        args = parser.parse_args()
        result = dialog_kobert.predict(args['s'])
        print('sentence:',args['s'] )
        print('result:',result )
        return make_response(json.dumps({'answer':result},ensure_ascii=False))

class DialogElectraAPI(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('s')
        args = parser.parse_args()
        result = dialog_electra.predict(args['s'])
        print('sentence:',args['s'] )
        print('result:',result )
        return make_response(json.dumps({'answer':result},ensure_ascii=False))

# class BertQaPost(Resource):
#
#     def post(self):
#         args = parser.parse_args() # context, question, answer로 구성
#
#         print(args['context'])
#         print(args['question'])
#         qa_example = {
#             'context':args['context'],
#             'question': args['question']
#         }
#         answer=qa_module.getAnswer([qa_example])
#         print(answer)
#         result = { 'result': answer}
#
#
#         return result, 201

api.add_resource(DialogKoBERTAPI,'/api/wellness/dialog/bert')
api.add_resource(DialogElectraAPI,'/api/wellness/dialog/electra')
# api.add_resource(BertQaPost,'/api/bert/qa/post')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9900)