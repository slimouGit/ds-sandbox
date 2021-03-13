import os
import hashlib

__author__      = "Salim Oussayfi"

counter = 1

def getInput():
    dir = input("Bitte einen Pfad angeben:\n")
    printDir(dir)

def printDir(dir):
    try:
        #value wird mit aktuellem Pfad initialisiert
        value = os.listdir(dir)
    except:
        #Eingabe ist kein gueltiges Verzeichnes
        print("\"" + dir + "\"" + " ist kein gültiger Pfad")
        getInput()
        return

    for item in value:
        counter+1
        print(counter)
        #falls Fund ein Ordner ist
        if os.path.isdir(dir+"/"+item) == True:
            newDir = dir+"/"+item #Pfad wird nach jedem Durchlauf neu generiert
            print("- Ordner:\t" + item + ", Pfad: " + dir)
            printDir(newDir)

        #falls Fund eine Datei ist
        if os.path.isfile(dir+"/"+item) == True:
            #MD5-Summe berechnen
            hash = hashlib.md5()
            hash.update(item.encode('utf-8'))
            hashNr = hash.hexdigest()
            print("- Datei:\t" + item + ", Pfad: " + dir + ", MD5: " + hashNr)

#Nutzereingabe wird gestartet
getInput()