# DNN-copula-loss
# DNN_copula_multivarie


# Remarque : SVP verifier que le dossier de la base de donnée data_copule est bien dans le dossier du code et aussi dans la racine .


#code inspired by https://github.com/lllllllllllll-llll/NSSADNN_IQA

le lien de l'environnement virtuel :

      https://drive.google.com/file/d/11r-jqV1ifbIX90br-oRsMV0q_H88TRn5/view?usp=sharing


2/Activez l'environnement virtuel à l'aide d'une commande spécifique à l'interface système :
   
      source ./cyclo/bin/activate


3/les bibliotheques necessaires :

      
      python version 3.6.9
      
      pip install torch==0.4.1

      pip install 'git+https://github.com/lanpa/tensorboardX'

      pip install Pillow==6.2.1

      pip install numpy==1.17.3

      pip install opencv-python

      pip install scipy==1.3.1

      pip install torchvision==0.2.2
     
      pip install PyYAML
      
      python -m pip install -U scikit-image
      
      pip install --pre torch torchvision -f                     https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
      
4/pour lancer le code, il vous suffit de lancer le fichier "python train.py"

apres le lancement du code, le nombre d'epoques continuera va etre affiché dans le cmd pour vous tenir au courant de l'evolution de l'execution. dans le code notre code, nous 
avons utilisé 500 epoques avec 586 iterations chacune.

5/ Pour quitter ensuite l'environnement virtualenv :
      
        deactivate
   
