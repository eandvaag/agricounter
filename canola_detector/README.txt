In order to use this canola seedling detection model in your own version of 
AgriCounter, you will need to follow these steps:

1. Sign in to the tool with the admin account.

2. Through the admin account interface, add a new object class called 
"Canola Seedling". (Click the "+" button next to the "Object Classes" heading 
to add a new object class.)

3. Through the admin account interface, create a user account. (Click the 
"+" button next to "User Accounts".)

4. Copy the canola_detector directory into the following location:
      backend/src/usr/data/<username>/models/available/public
where <username> is the username for the user account you created in step 3.

5. Edit the log.json file in the canola_detector directory. You will need to 
change the value of the "model_creator" field to "<username>".

Once you have completed these steps, you should be able to sign in with any 
user account and select the canola_detector model within the AgriCounter 
workspace interface.


If you are using this model for your own research, please cite our paper:

Erik Andvaag, Kaylie Krys, Steven J. Shirtliffe, Ian Stavness. Counting Canola: Toward Generalizable Aerial Plant Detection Models. Plant Phenomics. 2024;6:0268.DOI:10.34133/plantphenomics.0268
