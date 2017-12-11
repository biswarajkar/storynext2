# storyNext
## Installation
- Install MAMP https://www.mamp.info/en/
- Open MAMP
- Set root folder: Click preference -> Web Server -> Click Folder Icon ->select "web" folder in our repository (storynext2)
- Set CGI script running option:
  - PRO Version MAMP: File -> Edit -> Template -> Apache -> open up "httpd.conf"

  Search for "ScriptAlias" and replace the line with
  ```
ScriptAlias /cgi-bin/ "your-path-to-storynext2/web/cgi-bin/"
  ```

  Then search for `<Directory "/Applications/MAMP/cgi-bin">
  `
  And add a new (separate) block below it:
  ```
  <Directory "direct path to storynext2/web/cgi-bin/">
    Options ExecCGI
  AddHandler cgi-script .cgi
  </Directory>
  ```
  - Normal Version MAMP: Go to MAMP installed path -> conf ->apache ->open "httpd.conf" do the same setup
- run Server
- open home website
- insert your article
- click GO!