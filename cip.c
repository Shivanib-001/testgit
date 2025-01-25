 #include <sys/types.h>
 #include <sys/stat.h>
 #include <fcntl.h>
 #include <termios.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <unistd.h>
 #include <strings.h>
 #include <string.h>
 #include <curl/curl.h>


 /* baudrate settings are defined in <asm/termbits.h>, which is
 included by <termios.h> */
 #define BAUDRATE B9600
 /* change this definition for the correct port */
 #define MODEMDEVICE "/dev/ttyS0"
 #define _POSIX_SOURCE 1 /* POSIX compliant source */
 #define FALSE 0
 #define TRUE 1
 volatile int STOP=FALSE;
 /*--------------------- Function decleration ----------------------------*/
 void insert( int UTC_Time, float lat, float lng, float spd);
 double lat_filt(float def);
 double lng_filt(float kef);
 
 
 
 
 /*----------------------------------------------------------------------*/
 
 int main()
 {
 int fd,c, res;
 struct termios oldtio,newtio;
 char buf[255];
 char *token;
 int timk;
 float spd=0.0;
 float latitude=0.0, longitude=0.0;
 /*
 Open modem device for reading and writing and not as controlling tty
 because we don't want to get killed if linenoise sends CTRL−C.
 */
 fd = open(MODEMDEVICE, O_RDWR | O_NOCTTY );
 if (fd <0) {perror(MODEMDEVICE); exit(-1); }
 tcgetattr(fd,&oldtio); /* save current serial port settings */
 bzero(&newtio, sizeof(newtio)); /* clear struct for new port settings */
 /*
 BAUDRATE: Set bps rate. You could also use cfsetispeed and cfsetospeed.
 CRTSCTS : output hardware flow control (only used if the cable has
 all necessary lines. See sect. 7 of Serial−HOWTO)
 CS8 : 8n1 (8bit,no parity,1 stopbit)
 CLOCAL : local connection, no modem contol
 CREAD : enable receiving characters
 */
 newtio.c_cflag = BAUDRATE | CRTSCTS | CS8 | CLOCAL | CREAD;
 /*
 IGNPAR : ignore bytes with parity errors
 ICRNL : map CR to NL (otherwise a CR input on the other computer
 will not terminate input)
 otherwise make device raw (no other input processing)
 */
 newtio.c_iflag = IGNPAR | ICRNL;
 /*
 Raw output.
 */
 newtio.c_oflag = 0;
 /*
 ICANON : enable canonical input
 disable all echo functionality, and don't send signals to calling program
 */
 newtio.c_lflag = ICANON;
 /*
 initialize all control characters
 default values can be found in /usr/include/termios.h, and are given
 in the comments, but we don't need them here
 */
 newtio.c_cc[VINTR] = 0; /* Ctrl−c */
 newtio.c_cc[VQUIT] = 0; /* Ctrl−\ */
 newtio.c_cc[VERASE] = 0; /* del */
 newtio.c_cc[VKILL] = 0; /* @ */
 newtio.c_cc[VEOF] = 4; /* Ctrl−d */
 newtio.c_cc[VTIME] = 0; /* inter−character timer unused */
 newtio.c_cc[VMIN] = 1; /* blocking read until 1 character arrives */
 newtio.c_cc[VSWTC] = 0; /* '\0' */
 newtio.c_cc[VSTART] = 0; /* Ctrl−q */
 newtio.c_cc[VSTOP] = 0; /* Ctrl−s */
 newtio.c_cc[VSUSP] = 0; /* Ctrl−z */
 newtio.c_cc[VEOL] = 0; /* '\0' */
 newtio.c_cc[VREPRINT] = 0; /* Ctrl−r */
 newtio.c_cc[VDISCARD] = 0; /* Ctrl−u */
 newtio.c_cc[VWERASE] = 0; /* Ctrl−w */
 newtio.c_cc[VLNEXT] = 0; /* Ctrl−v */
 newtio.c_cc[VEOL2] = 0; /* '\0' */
 /*
 now clean the modem line and activate the settings for the port
 */
 tcflush(fd, TCIFLUSH);
 tcsetattr(fd,TCSANOW,&newtio);
 /*
 terminal settings done, now handle input
 In this example, inputting a 'z' at the beginning of a line will
 exit the program.
 */

 while (STOP==FALSE) { /* loop until we have a terminating condition */
 /* read blocks program execution until a line terminating character is
 input, even if more than 255 chars are input. If the number
 of characters read is smaller than the number of chars available,
 subsequent reads will return the remaining chars. res will be set
 to the actual number of characters actually read */
 res = read(fd,buf,255);
 token = strtok(buf,",");
 buf[res]=0; /* set end of string, so we can printf */
 int k=0;
 int rmc=0;
 int y=0;
// printf(":%s:%d\n", buf, res);
 do{
 // printf("token: %s %d %d \n",token,y,k);
  if(strcmp(token,"$GNRMC")==0){
   printf("GNRMC --- Found");
    rmc+=1;
    }

if(strcmp(token,"V") == 0 && rmc==1){
   printf("GNRMC --invalid");
}else if(strcmp(token,"V")!=0 && rmc==1){

y+=1;
}

if(k==y-1){

if(k==1){
//printf("Time : %d %d \n",atoi(token),y);
timk=atoi(token);
}
if(k==3){
//printf("lat: %f %d \n",atof(token),y);
//lat=atof(token);
latitude = lat_filt(atof(token));
}
if(k==5){
//printf("lng: %f %d \n",atof(token),y);
//lng=atof(token);
longitude= lng_filt(atof(token));
}
if(k==7){
//printf("speed: %f %d \n",atof(token),y);
spd=atof(token);
}
printf("%d %f %f %f \n",timk,latitude,longitude,spd);
/*------------------------------- ADD CURL FUNCTION ------------------------------------------*/
insert(timk,latitude,longitude,spd);
}

 k+=1;
}
while(token = strtok(NULL,","));

// if (buf[0]=='z') STOP=TRUE;
// usleep(1000);
 }
 /* restore the old port settings */
 tcsetattr(fd,TCSANOW,&oldtio);
 }
 
/*-------------------------------------------------------------------------------------------------------------------------------------*/ 
void insert( int UTC_Time, float lat, float lng, float spd) {
    unsigned int hour, min, sec;
    char time_str[9];  
    hour = (UTC_Time / 10000);
    min = (UTC_Time % 10000) / 100;
    sec = (UTC_Time % 10000) % 100;
    
    hour = hour + 5;  
    if (hour >= 24) {
        hour -= 24;  
    }
    
    min = min + 30;
    if (min > 59) {
        min -= 60;  
    }

    sprintf(time_str, "%02u:%02u:%02u", hour, min, sec);

    CURL *curl;
    CURLcode res;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if(curl) {
        const char *url = "https://ap-south-1.aws.data.mongodb-api.com/app/data-xnakx/endpoint/data/v1/action/insertOne";
        const char *api_key = "o87MBXtxmrtHBATmbQHxDrq2cYGmjxhfmh7szC8fn8C22qFJ2i1My4bGUEmtdJQi";
        char json_payload[1024];

        sprintf(json_payload, 
                "{"
                "\"dataSource\": \"Cluster0\","
                "\"database\": \"jan23\","
                "\"collection\": \"test3\","
                "\"document\": {"
                "\"time\": \"%s\","
                "\"latitude\": %.6f,"
                "\"longitude\": %.6f,"
                "\"speed\": %.2f"
                "}"
                "}", time_str, lat, lng, spd);

        struct curl_slist *headers = NULL;
        char api_key_header[256];
        sprintf(api_key_header, "api-key: %s", api_key);

        // Set headers for the request
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, api_key_header);

        // Set curl options
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload);

        // Perform the request
        res = curl_easy_perform(curl);

        if(res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            printf("Document inserted successfully!\n");
        }

        // Clean up
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    curl_global_cleanup();
}



double lat_filt(float def) {
	double latitude = 0.0;
	float k_lat_deg=(def*0.01);
	unsigned int deg = (int)k_lat_deg;
	if(deg > 8 && 37 > deg){
		float sec = (def- (float)deg*100)/60;
		latitude = (float)deg + sec;
	}

	return latitude;
}



double lng_filt(float kef){
	double longitude = 0.0;
	float k_lng_deg=(kef*0.01);
	unsigned int deglng = (int)k_lng_deg;
	if(deglng > 68 && 97 > deglng){
		float seclng = (kef- (float)deglng*100)/60;
		longitude = (float)deglng + seclng;

	}

	return longitude;
}


 
 
