from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from threading import Lock
from gevent import pywsgi
import geoip2.database
import geoip2.errors
import json
import os

async_mode = None
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
thread = None
thread_lock = Lock()

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'xls', 'JPG', 'PNG', 'xlsx', 'gif', 'GIF', 'json'}


# @app.route('/')
# def hello_world():
#     return 'Hello World!'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/generation', methods=['GET', 'POST'])
def generation():
    if request.method == "GET":
        '''初始化结果日志'''
        f = open('train.log', 'w')
        f.writelines([])
        f.close()
        f = open('generation.log', 'w')
        f.writelines([])
        f.close()
        return render_template('generation.html')


@app.route('/generation/upload', methods=['POST'], strict_slashes=False)
def generation_upload():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['myfile']  # 从表单的file字段获取文件，myfile为该表单的name值
    if f and allowed_file(f.filename):  # 判断是否是允许上传的文件类型
        fname = secure_filename(f.filename)
        # f.save(os.path.join(file_dir, fname))  # 保存文件到upload目录
        f.save("upload/data.txt")
        return render_template('generation.html', upload_return="上传成功")
    else:
        return render_template('generation.html', upload_return="上传失败")


@app.route("/generation/train", methods=['POST'])
def generation_train():
    forward_dict = request.form.to_dict()
    print(forward_dict)
    if forward_dict["model"] == "6GAN":
        os.system("cp upload/data.txt generation_model/6GAN/data/source_data/responsive-addresses.txt")
        os.system("python generation_model/6GAN/train.py > train.log 2>&1")
    elif forward_dict["model"] == "6VecLM":
        os.system("cp upload/data.txt generation_model/6VecLM/data/public_dataset/sample_addresses.txt")
        os.system("python generation_model/6VecLM/data_processing.py > train.log 2>&1")
        os.system("python generation_model/6VecLM/ipv62vec.py >> train.log 2>&1")
        os.system("python generation_model/6VecLM/ipv6_transformer.py >> train.log 2>&1")
    elif forward_dict["model"] == "6GCVAE":
        os.system("cp upload/data.txt generation_model/6GCVAE/data/public_datasets/responsive-addresses.txt")
        os.system("python generation_model/6GCVAE/data_process.py > train.log 2>&1")
        os.system("python generation_model/6GCVAE/gcnn_vae.py >> train.log 2>&1")

    return render_template('generation.html')


@app.route('/generation/query', methods=['GET', 'POST'])
def query():
    f = open('train.log', 'r')
    log = f.read()
    f.close()
    return log


@app.route('/generation/query_gen', methods=['GET', 'POST'])
def query_gen():
    f = open('generation.log', 'r')
    log = f.read()
    f.close()
    return log


@app.route('/generation/gen', methods=['POST'])
def generation_parameter():
    parameters = request.form.to_dict()
    print('parameters', parameters)
    number = parameters['number']
    model_name = parameters['model']
    if model_name == "6GAN":
        write_addresses = []
        f = open("generation_model/6GAN/candidate_set/candidate_generator_1_epoch_10.txt", "r")
        write_addresses += f.readlines()[:number / 6]
        f.close()
        f = open("generation_model/6GAN/candidate_set/candidate_generator_2_epoch_10.txt", "r")
        write_addresses += f.readlines()[:number / 6]
        f.close()
        f = open("generation_model/6GAN/candidate_set/candidate_generator_3_epoch_10.txt", "r")
        write_addresses += f.readlines()[:number / 6]
        f.close()
        f = open("generation_model/6GAN/candidate_set/candidate_generator_4_epoch_10.txt", "r")
        write_addresses += f.readlines()[:number / 6]
        f.close()
        f = open("generation_model/6GAN/candidate_set/candidate_generator_5_epoch_10.txt", "r")
        write_addresses += f.readlines()[:number / 6]
        f.close()
        f = open("generation_model/6GAN/candidate_set/candidate_generator_6_epoch_10.txt", "r")
        write_addresses += f.readlines()[:number / 6]
        f.close()
        f = open("download/output.txt", "w")
        f.writelines(write_addresses)
        f.close()
        f = open("generation.log", "w")
        f.write("执行完成！")
        f.close()
    elif model_name == "6VecLM":
        f = open("generation_model/6VecLM/model_load.py", "r")
        codes = f.readlines()
        f.close()
        codes[13] = "train_data_size = " + number + "\n"
        f = open("generation_model/6VecLM/model_load.py", "w")
        f.writelines(codes)
        f.close()
        os.system("python generation_model/6VecLM/model_load.py > generation.log 2>&1")
        os.system("cp generation_model/6VecLM/data/generation_data/candidate_s6_e1_t0.1.txt download/output.txt")
    elif model_name == "6GCVAE":
        f = open("generation_model/6GCVAE/generation.py", "r")
        codes = f.readlines()
        f.close()
        codes[10] = "generation_number = " + number + "\n"
        f = open("generation_model/6GCVAE/generation.py", "w")
        f.writelines(codes)
        f.close()
        os.system("python generation_model/6GCVAE/generation.py > generation.log 2>&1")
        os.system("cp generation_model/6GCVAE/data/generated_data/6gcvae_generation.txt download/output.txt")

    return render_template('generation.html')


@app.route('/generation/download', methods=['POST'])
def generation_download():
    path = "download/output.txt"
    return send_from_directory(app.root_path, filename=path, as_attachment=True)


@app.route('/linkage', methods=['GET', 'POST'])
def linkage():
    if request.method == "GET":
        '''初始化结果日志'''
        f = open('train2.log', 'w')
        f.writelines([])
        f.close()
        return render_template('linkage.html')


@app.route('/linkage/upload', methods=['POST'], strict_slashes=False)
def linkage_upload():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['myfile']  # 从表单的file字段获取文件，myfile为该表单的name值
    if f and allowed_file(f.filename):  # 判断是否是允许上传的文件类型
        fname = secure_filename(f.filename)
        # f.save(os.path.join(file_dir, fname))  # 保存文件到upload目录
        f.save("upload/data.json")
        return render_template('linkage.html', upload_return="上传成功")
    else:
        return render_template('linkage.html', upload_return="上传失败")


@app.route("/linkage/train", methods=['POST'])
def linkage_train():
    forward_dict = request.form.to_dict()
    print(forward_dict)
    if forward_dict["model"] == "SiamHAN":
        os.system("cp upload/data.json linkage_model/SiamHAN/data/cstnet.json")
        os.system("python linkage_model/SiamHAN/train.py > train2.log 2>&1")
        os.system("zip -r download/output.zip linkage_model/SiamHAN/pre_trained/")
    elif forward_dict["model"] == "GALG":
        os.system("cp upload/data.json linkage_model/GALG/data/cstnet.json")
        os.system("python linkage_model/GALG/main.py > train2.log 2>&1")
        os.system("zip -r download/output.zip linkage_model/GALG/models/")
    return render_template('linkage.html')


@app.route('/linkage/query_link', methods=['GET', 'POST'])
def query_link():
    f = open('train2.log', 'r')
    log = f.read()
    f.close()
    return log


@app.route('/linkage/download', methods=['POST'])
def linkage_download():
    path = "download/output.zip"
    return send_from_directory(app.root_path, filename=path, as_attachment=True)


@app.route('/gen-analysis', methods=['GET', 'POST'])
def gen_analysis():
    if request.method == "GET":
        return render_template('gen-analysis.html')


@app.route('/gen-analysis/upload', methods=['POST'], strict_slashes=False)
def gen_analysis_upload():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['myfile']  # 从表单的file字段获取文件，myfile为该表单的name值
    if f and allowed_file(f.filename):  # 判断是否是允许上传的文件类型
        fname = secure_filename(f.filename)
        # f.save(os.path.join(file_dir, fname))  # 保存文件到upload目录
        address_path = "upload/candidate.txt"
        f.save(address_path)
        return render_template('gen-analysis.html', upload_return="上传成功")
    else:
        return render_template('gen-analysis.html', upload_return="上传失败")


@app.route("/gen-analysis/analyse", methods=['GET', 'POST'])
def address_analyse():
    address_path = "upload/candidate.txt"
    max_num = 20
    f = open(address_path, 'r')
    address_list = f.readlines()
    f.close()
    addresses = []
    for address in address_list:
        addresses.append(address[:-1])

    asn_dict = {}
    country_code_dict = {}
    asn_description_dict = {}
    network_dict = {}
    city_dict = {}
    geo_dict = {}
    city_reader = geoip2.database.Reader("GeoLite2-City.mmdb")
    asn_reader = geoip2.database.Reader("GeoLite2-ASN.mmdb")
    for address in addresses:
        city_response = None
        asn_response = None
        try:
            city_response = city_reader.city(address)
            asn_response = asn_reader.asn(address)
        except geoip2.errors.AddressNotFoundError:
            pass
        if city_response is None or asn_response is None:
            continue
        # print(city_response.raw)
        # print(asn_response.raw)

        asn = asn_response.autonomous_system_number
        country_code = city_response.country.iso_code
        asn_description = asn_response.autonomous_system_organization
        network = str(city_response.traits.network)
        city = city_response.city.name

        asn_dict[asn] = 1 if asn not in asn_dict.keys() else asn_dict[asn] + 1
        country_code_dict[country_code] = 1 if country_code not in country_code_dict.keys() else \
            country_code_dict[country_code] + 1
        asn_description_dict[asn_description] = 1 if asn_description not in asn_description_dict.keys() else \
            asn_description_dict[asn_description] + 1
        network_dict[network] = 1 if network not in network_dict.keys() else network_dict[network] + 1
        if city is not None:
            city_dict[city] = 1 if city not in city_dict.keys() else city_dict[city] + 1
            city_name = city_response.city.names["zh-CN"] if "zh-CN" in city_response.city.names.keys() else city_response.city.name
            if city_name not in geo_dict.keys():
                geo_dict[city_name] = {}
                geo_dict[city_name]["value"] = 1
                geo_dict[city_name]["latitude"] = city_response.location.latitude
                geo_dict[city_name]["longitude"] = city_response.location.longitude
            else:
                geo_dict[city_name]["value"] += 1
        # else:
        #     country_name = city_response.country.names["zh-CN"] if "zh-CN" in city_response.country.names.keys() else city_response.country.name
        #     if country_name not in geo_dict.keys():
        #         geo_dict[country_name] = {}
        #         geo_dict[country_name]["value"] = 1
        #         geo_dict[country_name]["latitude"] = city_response.location.latitude
        #         geo_dict[country_name]["longitude"] = city_response.location.longitude
        #     else:
        #         geo_dict[country_name]["value"] += 1

    asn_dict_s = sorted(asn_dict.items(), key=lambda x: x[1], reverse=True)
    country_code_dict_s = sorted(country_code_dict.items(), key=lambda x: x[1], reverse=True)
    asn_description_dict_s = sorted(asn_description_dict.items(), key=lambda x: x[1], reverse=True)
    network_dict_s = sorted(network_dict.items(), key=lambda x: x[1], reverse=True)
    city_dict_s = sorted(city_dict.items(), key=lambda x: x[1], reverse=True)

    asn_x = []
    asn_y = []
    for i in range(min(len(asn_dict_s), max_num)):
        asn_x.append(asn_dict_s[i][0])
        asn_y.append(asn_dict_s[i][1])

    country_code = []
    for i in range(min(len(country_code_dict_s), max_num)):
        temp_dict = {}
        temp_dict["value"] = country_code_dict_s[i][1]
        temp_dict["name"] = country_code_dict_s[i][0]
        country_code.append(temp_dict)

    asn_description_x = []
    asn_description_y = []
    for i in range(min(len(asn_description_dict_s), max_num)):
        asn_description_x.append(asn_description_dict_s[i][0])
        asn_description_y.append(asn_description_dict_s[i][1])

    network_x = []
    network_y = []
    for i in range(min(len(network_dict_s), max_num)):
        network_x.append(network_dict_s[i][0])
        network_y.append(network_dict_s[i][1])

    city = []
    for i in range(min(len(city_dict_s), max_num)):
        temp_dict = {}
        temp_dict["value"] = city_dict_s[i][1]
        temp_dict["name"] = city_dict_s[i][0]
        city.append(temp_dict)

    geo = {}
    geo_value = []
    for key, values in geo_dict.items():
        geo_value.append({"name": key, "value": values["value"]})
        geo[key] = [values["longitude"], values["latitude"]]
    print(geo)
    print(geo_value)
    print(len(geo))

    return render_template('gen-analysis.html', city=city,
                                                asn_x=asn_x, asn_y=asn_y,
                                                asn_country_code=country_code,
                                                asn_description_x=asn_description_x, asn_description_y=asn_description_y,
                                                network_x=network_x, network_y=network_y,
                                                geo_value=geo_value, geo=geo)


@app.route('/link-analysis', methods=['GET', 'POST'])
def link_analysis():
    if request.method == "GET":
        '''初始化结果日志'''
        f = open('tracking.log', 'w')
        f.writelines([])
        f.close()
        f = open('discovery.log', 'w')
        f.writelines([])
        f.close()
        return render_template('link-analysis.html')


@app.route('/link-analysis/upload', methods=['POST'], strict_slashes=False)
def link_analysis_upload():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['myfile']  # 从表单的file字段获取文件，myfile为该表单的name值
    if f and allowed_file(f.filename):  # 判断是否是允许上传的文件类型
        fname = secure_filename(f.filename)
        # f.save(os.path.join(file_dir, fname))  # 保存文件到upload目录
        address_path = "upload/data.json"
        f.save(address_path)
        return render_template('link-analysis.html', upload_return="上传成功")
    else:
        return render_template('link-analysis.html', upload_return="上传失败")


@app.route('/link-analysis/draw_link_dataset', methods=['GET', 'POST'], strict_slashes=False)
def draw_link_dataset():
    address_path = "upload/data.json"
    dataset = []
    f = open(address_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()

    json_string = ""
    for line in lines:
        if line[:-1] == '{':
            json_string = ""
        json_string += line[:-1]
        if line[:-1] == '}':
            dataset.append(json.loads(json_string))

    new_json = {}
    new_json["nodes"] = []
    new_json["links"] = []
    new_json["categories"] = []
    id = 0
    category = 0
    for _, user_data in enumerate(dataset):
        if _ == 50:
            break
        if int(user_data['nodes']['node_count']) > 1:
            for i in range(int(user_data['nodes']['node_count'])):
                temp = {}
                temp["id"] = id
                temp["name"] = user_data['nodes']['node_' + str(i)]['client_features']['ip']
                temp["symbolSize"] = int(user_data['nodes']['node_count'])
                temp["value"] = int(user_data['nodes']['node_count'])
                temp["category"] = "User " + str(category)
                new_json["nodes"].append(temp)
                for j in range(id + 1, id + int(user_data['nodes']['node_count'] - i)):
                    new_json["links"].append({"source": id, "target": j})
                id += 1
            new_json["categories"].append({"name": "User " + str(category)})
            category += 1
    # with open("draw_link.json", "w", encoding="ISO-8859-1") as f:
    #     json.dump(new_json, f, indent=4, separators=(',', ': '))
    return jsonify(new_json)


@app.route('/link-analysis/query_tracking', methods=['GET', 'POST'])
def query_tracking():
    f = open('tracking.log', 'r')
    log = f.read()
    f.close()
    return log


@app.route("/link-analysis/tracking", methods=['GET', 'POST'])
def link_analysis_tracking():
    os.system("cp upload/data.json linkage_model/SiamHAN/data/cstnet.json")
    os.system("python linkage_model/SiamHAN/inference_tracking_re.py > tracking.log 2>&1")
    return render_template('link-analysis.html')


@app.route('/link-analysis/query_discovery', methods=['GET', 'POST'])
def query_discovery():
    f = open('discovery.log', 'r')
    log = f.read()
    f.close()
    return log


@app.route("/link-analysis/discovery", methods=['GET', 'POST'])
def link_analysis_discovery():
    os.system("cp upload/data.json linkage_model/SiamHAN/data/cstnet.json")
    os.system("python linkage_model/SiamHAN/inference_discovery_re.py > discovery.log 2>&1")
    return render_template('link-analysis.html')


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=60000)
    # server = pywsgi.WSGIServer(('127.0.0.1', 60000), app)
    # server.serve_forever()
