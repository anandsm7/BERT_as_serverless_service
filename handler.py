import json
import config as cfg
import inference
import logging

infer = inference.Inference(cfg.MODEL_PATH, cfg.DEVICE)


def predict(event, context):
    try:
        body = json.loads(event['body'])
        context.log(body)
        preds = infer.sentence_prediction(sentence=body['log'])
        context.log(preds)
        logging.info(f"prediction: {preds}")

        return {
            "statusCode": 200,
            "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
            "body": json.dumps({"prediction": preds['class']})
        }
    except Exception as e:
        logging.error(e)
        return {
            "statusCode": 500,
            "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
            "body": json.dumps({"error": repr(e)})
        }
