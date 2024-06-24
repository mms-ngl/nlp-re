## Relation Extraction

Relation Extraction (RE) project for "Natural Language Processing" course.

### 📝 Project documentation

[**REPORT**](https://github.com/mms-ngl/nlp-re/blob/main/report.pdf)

### Course Info: http://naviglinlp.blogspot.com/

### 🚀 Project setup

#### Project directory
[[Downloads]](https://drive.google.com/drive/folders/15zehfVqaSfWc4k8bogBa6VhhEW0rGopi?usp=sharing) Trained models and their associated files in er and rc folders: config.json, pytorch_model.bin, special_tokens_map.json, tokenizer.json, tokenizer_config.json, vocab.txtlabel2id.pth, model_weights.pth.
```
root
- data
- logs
- model
 - er
  - config.json
  - pytorch_model.bin
  - special_tokens_map.json
  - tokenizer.json
  - tokenizer_config.json
  - vocab.txt
 - rc
  - config.json
  - pytorch_model.bin
  - special_tokens_map.json
  - tokenizer.json
  - tokenizer_config.json
  - vocab.txt 
 - .placeholder
- re
- Dockerfile
- README
- report
- requirements
- test
```

#### Requirements

* Ubuntu distribution
  * Either 20.04 or the current LTS (22.04).
* Conda 

#### Setup Environment

To run *test.sh*, we need to perform two additional steps:

* Install Docker
* Setup a client

#### Install Docker

```bash
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

#### Setup Client

```bash
conda create -n nlp-re python=3.9
conda activate nlp-re
pip install -r requirements.txt
```

#### Run

*test.sh* is a simple bash script. To run it:

```bash
conda activate nlp-re
bash test.sh data/test.jsonl
```
