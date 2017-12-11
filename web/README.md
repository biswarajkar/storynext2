## Web Application Server Setup
**We have used MAMP to run our UI on a server locally, however, any CGI supported Application Server (or cloud hosting) can also be used to host our application.**
## Steps:
- Install MAMP from https://www.mamp.info/en/
- Open MAMP
- Set root folder: Click preference -> Web Server -> Click Folder Icon ->select "web" folder in our repository (storynext2) 
  
Set CGI script running option as follows:
### PRO Version MAMP: 
- Go to the menu option File -> Edit -> Template -> Apache -> Open up "httpd.conf"    
- Search for "ScriptAlias" and replace the line with
  ``ScriptAlias /cgi-bin/ "your-path-to-storynext2/web/cgi-bin/" ``
- Then search for ``<Directory "/Applications/MAMP/cgi-bin">`` and add a new (separate) block below it:     
  `  <Directory "direct path to storynext2/web/cgi-bin/">    `     
       `Options ExecCGI    `       
       `AddHandler cgi-script .cgi    `      
  `  </Directory>   `

### Normal Version MAMP: 
- Go to MAMP installed path -> conf ->apache ->open "httpd.conf" do the same setup
- Run Server
- Open StoryNext Homepage (index.html)
- Insert/type in your article
- Click on GO!
