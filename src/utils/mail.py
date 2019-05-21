import smtplib

username = "bettix4@gmail.com"
password = ""
mittente = username
desinatario = username

oggetto = "Subject: Urgente! da leggere subito!\n\n"
contenuto = "connettiti al Server che è meglio..."
messaggio = oggetto + contenuto

email = smtplib.SMTP("smtp.gmail.com", 587)
email.ehlo()
email.starttls()
email.login(username, password)
email.sendmail(mittente, desinatario, messaggio)
email.quit()
