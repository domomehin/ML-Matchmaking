# import pymongo
from pymongo import MongoClient

conn_str = "mongodb+srv://kimia:EaoHrgfNcva27lMT@cluster0.6ihoy.mongodb.net/match-it?retryWrites=true&w=majority"
# conn_str = "mongodb+srv://<username>:<password>@<cluster-address>/test?retryWrites=true&w=majority"

client = MongoClient(conn_str)

def get_stylists():
    collection_stylist = client.get_database("match-it").get_collection("stylist")
    documents_stylist = collection_stylist.find({})
    information = []
    for document in documents_stylist:
        stylist = {}
        stylist['interests'] = document['interests']
        stylist['rate'] = document['age']
        stylist['id'] = document['_id']
        information.append(stylist)
    return information

    
    
def get_styleseekers():
    collection_style_seeker = client.get_database("match-it").get_collection("styleseeker")
    documents_style_seeker = collection_style_seeker.find({})
    interests = []
    for document in documents_style_seeker:
        interests.append(document['interests'])
    return interests

def get_stylist_from_id(id):
    collection_stylist = client.get_database("match-it").get_collection("stylist")
    documents_stylist = collection_stylist.find({"_id": id})
    ans = None
    for doc in documents_stylist:
        ans = doc
    return ans
    
## interests
## ["18-29 group", "$15-$20", "funky", ""]