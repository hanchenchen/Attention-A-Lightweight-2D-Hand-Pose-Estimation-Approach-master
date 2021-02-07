import argparse
import json
import os
from model_train import create_model
import frei_hand
import tensorflow as tf
parser = argparse.ArgumentParser(description='use the specified dataset to train the model.')
parser.add_argument('dataset_name', type=str, default='FreiHAND_pub_v2',
                    help='choose one dataset.')
args = parser.parse_args()
choosed = ['FreiHAND_pub_v2', 'Panoptic']
f = open('configs/' + args.dataset_name + '.json')
configs = json.load(f)
f.close()
images_path = tf.io.gfile.glob(configs['images_path'])
num = len(images_path)
testing = images_path[-int(num*0.1):]
tf.print(testing[0])
# testing_images, ground_truth = frei_hand.get_testing_dataset(testing)
filepath = args.dataset_name + '/weights.hdf5'
model = create_model()
model.load_weights(filepath)  # continue to train
tf.print(model[frei_hand.load_image(testing[0])])
'''
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data...")
evaluate_fpath = open(os.path.join(args.dataset_name, 'evaluate.json'), 'w')
results = json.load(evaluate_fpath)
results[configs['learning_rate']] = model.evaluate(testing_images, ground_truth)
print("test loss, test acc:", results)
json.dump(results, evaluate_fpath)
evaluate_fpath.close()


# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions: ", testing_images.shape)
predictions = model.predict(testing_images)#, steps = 100)
# print("predictions shape:", predictions.shape)
# tf.print(predictions)

# Save the predictions to (args.dataset_name)/predictions.json
predictions_fpath = open(os.path.join(args.dataset_name, 'predictions.json'), 'w')
pred_list = [i.tolist() for i in predictions]
pred_list = {testing[i]:pred_list[i] for i in range(len(testing))}
json.dump(pred_list, predictions_fpath)
predictions_fpath.close()
'''