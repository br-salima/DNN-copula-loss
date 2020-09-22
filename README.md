# NR-IQA-copula-features-as-auxiliary-ask
#code inspired by https://github.com/lllllllllllll-llll/NSSADNN_IQA

1/ Créer un environnement virtuel 
   
   	virtualenv venv
2/Activez l'environnement virtuel à l'aide d'une commande spécifique à l'interface système :
   
   	source ./venv/bin/activate


3/les bibliotheques necessaires :

       python version 3.6.9
      pip install torch==0.4.1

      pip install 'git+https://github.com/lanpa/tensorboardX'

      pip install Pillow==6.2.1

      pip install Pillow==6.2.1

      pip install numpy==1.17.3

      pip install opencv-python

      pip install scipy==1.3.1

      pip install torchvision==0.2.2

4/pour lancer le code, il vous suffit de lancer le fichier "python train.py"

apres le lancement du code, le nombre d'epoques continuera va etre affiché dans le cmd pour vous tenir au courant de l'evolution de l'execution. dans le code notre code, nous 
avons utilisé 500 epoques avec 586 iterations chacune.

5/ Pour quitter ensuite l'environnement virtualenv :
   
   	deactivate


l'architecture du modele :
![alt text](https://github.com/br-salima/NR-IQA-copula-features-as-auxiliary-ask/blob/master/model_with_CNNfeatureExtractor.PNG?raw=true)



	
