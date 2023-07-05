
# 此功能等同于trtexec，都是通过onnx输出engine，可用于开发中通过相关trt的api调用转换为engine模型。

# 导入必用依赖
import tensorrt as trt
onnx_path='/home/donggua/文档/yolov5-6.2/weights/yolov5s.onnx'
engine_path='/home/donggua/文档/yolov5-6.2/weights/yolov5s_16.engine'
# 创建logger：日志记录器
logger = trt.Logger(trt.Logger.WARNING)
 
# 创建构建器builder
builder = trt.Builder(logger)
# 预创建网络
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
# 加载onnx解析器
parser = trt.OnnxParser(network, logger)
success = parser.parse_from_file(onnx_path)
for idx in range(parser.num_errors):
  print(parser.get_error(idx))
if not success:
  print("onnx2trt failed!")
  pass  # Error handling code here
# builder配置
config = builder.create_builder_config()
# 分配显存作为工作区间，一般建议为显存一半的大小
config.max_workspace_size = 1 << 30  # 1 Mi
config.set_flag(trt.BuilderFlag.FP16)
serialized_engine = builder.build_serialized_network(network, config)
# 序列化生成engine文件
with open(engine_path, "wb") as f:
   f.write(serialized_engine)
   print("generate file success!")