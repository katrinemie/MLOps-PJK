Exercises/Tasks 5:

You’ve just started your role as an MLOps engineer at a company when an urgent task lands on your desk. A developer from the team has trained a promising neural network for image classification, and your boss is so impressed that he wants to demo it to an important customer tomorrow. The catch? The demo needs to run on his smartphone, and it must be fast. That means you’ll need to quantize the model before deployment.

However, the development team has no experience in deploying trained models, so it’s up to you to make it happen. This means that you must put your current MLOps project on hold. Your boss drops his Android phone on your desk and forwards an email with details about the model. Since the project is classified, the only information provided is that the model is in ONNX format with FP32 precision, and it can be downloaded from the link here. 

Time to put your MLOps skills to the test! 

You will loan one Samsung Galaxy phone (already in developer mode) per group, which you can have for up to two weeks. Since the course focus is on the MLOps part, and not UI development, you are recommended to start using the Hint 1 below. If you do, this is what the demo of the secret model could look like on a Galaxy S21 Ultra:

Hint 1: 
Follow this guide to deploy a MobileNet-based classification model on Android: https://github.com/microsoft/onnxruntime-inference-examples/tree/main/mobile/examples/image_classification/android

Hint 2:
Quantize the secret model from FP32 to UInt8 precision using the ONNX runtime Python library: 

from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic

model_fp32 = 'secret_model.onnx'
model_quant = 'secret_model_int8.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)


Hint 3:
Copy the FP32, and quantized model, to the folder:  onnxruntime-inference-examples/mobile/examples/image_classification/android/app/src/main/res/raw/

Replace the mobilenetv2_int8 and mobilenetv2_fp32 models in the readModel() function in the MainActivity.kt source file with the FP32 and quantized versions of the secret model 

Secret model info
The secret model is the Resnet50-v1-12.onnx from here https://github.com/onnx/models/tree/main/validated/vision/classification/resnet/model


Documentation

In addition to reflecting on the topics covered in this lecture and explaining how you have applied specific methods within your MLOps project, you must also document the following in your report.

D5.1: What was the inference time of the secret model on the phone before and after quantization? Document with a picture of the running demo.
D5.2: Explain how you have/or could have tested the endpoints related to your specific model (functional, robustness, performance, and security), and safeguarded against undesired input/output. What is important in your case and why?