/*
Author: Hena Naaz
Class: ECE6122
Last Date Modified: 20 Nov
Description:
The purpose of this file is to create a server for TCP connections.
*/

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <fstream>
#include <vector>
#include <utility>

using namespace std;

vector< pair<int, string> > allClientsInfo;

int n = 0;
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;

std::ofstream fout;


/*
    return time and date in the asked format
*/
char* getDateAndTime() {
    time_t now = time(0);
    char* dt = ctime(&now);
    dt[strlen(dt)-1] = '\0';

    return dt;
}


/*
    Store messages sent by the clients into the server.log file
*/
void saveMessage(string msg, pair<int, string> &clientInfo)
{
    /*
        taking lock so that second thread don't overwrite on the previous thread.
    */
    pthread_mutex_lock(&mutex1);
    fout.open("./server.log", std::ios_base::app);
    fout<<getDateAndTime()<<" :: "<<clientInfo.second<<" :: "<<msg;
    fout.close();
    pthread_mutex_unlock(&mutex1);
}


/*
    this function recieves the message sent by clients.
*/
void *recieveMessage(void *temp)
{
    pair<int, string> clientInfo = allClientsInfo.back();
    
    char message[500];

    for(int i=0; i<500; i++) {
        message[i] = '\0';
    }

    while(recv(clientInfo.first, message, 500, 0) > 0) {
        saveMessage(message, clientInfo);
        for(int i=0; i<500; i++) {
            message[i] = '\0';
        }
    }

    /*
        if any client quit, then removing that client from the client 
        directory maintained by server.
    */
    pthread_mutex_lock(&mutex1);

    fout.open("./server.log", std::ios_base::app);
    fout<<getDateAndTime()<<" :: "<<clientInfo.second<<" :: "<<"Disconnected\n";
    fout.close();

    for(int i=0; i<allClientsInfo.size(); i++) {
        if(allClientsInfo[i].first == clientInfo.first) {
            allClientsInfo.erase(allClientsInfo.begin()+i);
            break;
        }
    }

    pthread_mutex_unlock(&mutex1);

    return NULL;
}

/*
    check the port number passed in the console argument is valid or not
    and return port number if it valid else it will return 0

    @return int
*/
int getPortNumber(char* portNumber) {
    for(int i=0; portNumber[i] != '\0'; i++) {
        if(portNumber[i]<'0' || portNumber[i]>'9') {
            return 0;
        }
    }

    return atoi(portNumber);
}


/*
    this function adds client to the server directory.
*/
bool addClient(int clientSocket, string ipAddress) {
    allClientsInfo.push_back(make_pair(clientSocket, ipAddress));

    fout.open("./server.log", std::ios_base::app);
    fout<<getDateAndTime()<<" :: "<<ipAddress<<" :: "<<"Connected\n";
    fout.close();

    return true;
}

/*
    This is the driver function
*/
int main(int argc,char *argv[])
{

    struct sockaddr_in serverAddress, clientAddress;
    int serverSocket, clientSocket, portNumber;
    socklen_t clientAddressSize;
    pthread_t recieveThread;
    char ipAddress[INET_ADDRSTRLEN];

    if(argc > 2) {
        printf("Invalid command line argument detected: \nPlease check your values and press any key to end the program!\n");
        exit(1);
    }

    portNumber = getPortNumber(argv[1]);

    if( portNumber < 61000 || portNumber > 65535) {
        printf("Invalid command line argument detected: %s \nPlease check your values and press any key to end the program!\n", argv[1]);
        exit(1);
    }


    serverSocket = socket(AF_INET,SOCK_STREAM,0);

    for(int i=0; i<8; i++) {
        serverAddress.sin_zero[i] = '\0';
    }

    // memset(serverAddress.sin_zero,'\0',sizeof(serverAddress.sin_zero));

    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(portNumber);
    serverAddress.sin_addr.s_addr = inet_addr("127.0.0.1");
    clientAddressSize = sizeof(clientAddress);

    if(bind(serverSocket,( struct sockaddr *) &serverAddress,sizeof(serverAddress)) != 0 || listen(serverSocket, 5) != 0) {
        printf("Server is unable to bind/listen");
        exit(1);
    }

    while(1) {
        clientSocket = accept(serverSocket,(struct sockaddr *)&clientAddress,&clientAddressSize);
        
        if(clientSocket < 0) {
            printf("accept unsuccessful");
            exit(1);
        }

        pthread_mutex_lock(&mutex1);
        inet_ntop(AF_INET, (struct sockaddr *) &clientAddress, ipAddress, INET_ADDRSTRLEN);        
        addClient(clientSocket, ipAddress);
        pthread_create(&recieveThread, NULL, recieveMessage, NULL);
        pthread_mutex_unlock(&mutex1);
    }

    return 0;
}



















