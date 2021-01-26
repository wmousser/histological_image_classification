
### For Breast cancer :
Download the New BreakHis Database from 
https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

To run the program please :

Change the config.yml : 
1. Set the breakHis path into "dataset_path" value
2. Edit the list of "classes" to use for the classification
3. Set the "magnification" value
4. 



### For Cervix cancer :


Download the New Pap-smear Database from http://mde-lab.aegean.gr/downloads
- Remove white space in data set folder (New_database_pictures)
(todo: write a script) Run it 
  


- Generate experiment data set using : scripts/generate_dataset.sh
- Balance the data using balance_data.py module
- Train the model using : scripts/train_model.sh

Dependencies
1. Create a virtual environment then install required packages using the command : 
   
pip install requirements.txt   
+++++++++++++++++
2. Algorithm
1. Load data set
2. If new run : 
- root = Node()
- cl0 & cl1 = choose randomly 2
- build clf
- put all in root
- train & save gan for cl1 & cl0
2. Else : # if ! new run ...
- root = Load previous data structure if exists
- for cl in batch
    - pred_cl, node, position = root.get_node_to_update(cl_x_train)
    - new_node = Node()
    - fill in attributes by appending cl into position_classes
    - build clf(x_train = cl_x_train + gan(pred_cl)) 
    - right & left = None
    - node.position = new_node
    - node.position_classes.append(cl)
    - train & save à gan for cl