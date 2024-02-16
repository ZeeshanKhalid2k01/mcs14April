from flask import Flask, jsonify, request


# # Your code goes her
# import pandas as pd
# import numpy as np
# import os
# import warnings
import pickle
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# import xgboost as xgb
# import os
# from tensorflow.keras.models import load_model
# from urllib.parse import urlparse
# import os
# import pandas as pd
# import pickle,os
# import pandas as pd
# from urllib.parse import urlparse
# import re
# import time

from urllib.parse import urlparse
import os
import pandas as pd
from urllib.parse import urlparse
import re
app = Flask(__name__)

class Predict():
    def __init__(self):
        with open(os.path.join('models','random_forest_model.pkl'), 'rb') as f:
            self.rf_model = pickle.load(f)
        with open(os.path.join('models','gbdt_model.pkl'), 'rb') as f:
            self.gbdt_model = pickle.load(f)
        with open(os.path.join('models','lgb_model.pkl'), 'rb') as f:
            self.lgb_model = pickle.load(f)
        with open(os.path.join('models','xgboost_model.pkl'), 'rb') as f:
            self.xgb_model_2 = pickle.load(f)
        # self.lb_make = LabelEncoder()
        # self.xgb_model = xgb.Booster()
        # self.xgb_model.load_model(os.path.join('models','xgboost_model_cookie.model'))
        # # load the saved LSTM model
        # self.lstm_model = load_model(os.path.join("models","lstm_model.h5"))

        # # load the saved CNN model
        # self.cnn_model = load_model(os.path.join("models","CNN_model.h5"))
        self.suspicious = 0
    def pred(self, dict):
        self.dict = dict
        self.suspicious = 0
        self.df, self.df_c = self.process_input_dict()
        # self.df_c['Accept_Header_Length'] = self.df_c['AcceptHdr'].apply(lambda i: len(str(i)))
        # self.df_c['Accept_Header_SubDirectory'] = self.df_c['AcceptHdr'].apply(lambda i: self.no_of_dir(i))
        # # Length of URL
        # self.df_c['Cookie_Length'] = self.df_c['Cookie'].apply(lambda i: len(str(i)))
        # self.df_c['cookie_less_than_count'] = self.df_c['Cookie'].apply(lambda i: i.count('<'))
        # self.df_c['cookie_open_brace_count'] = self.df_c['Cookie'].apply(lambda i: i.count('{'))
        # self.df_c['cookie_close_brace_count'] = self.df_c['Cookie'].apply(lambda i: i.count('}'))
        # self.df_c['cookie_plus_count'] = self.df_c['Cookie'].apply(lambda i: i.count('+'))
        # self.df_c['cookie_minus_count'] = self.df_c['Cookie'].apply(lambda i: i.count('-'))
        # self.df_c['cookie_double_quote_count'] = self.df_c['Cookie'].apply(lambda i: i.count('"'))
        # self.df_c['cookie_colon_count'] = self.df_c['Cookie'].apply(lambda i: i.count(':'))
        # self.df_c['cookie_semicolon_count'] = self.df_c['Cookie'].apply(lambda i: i.count(';'))
        # self.df_c['cookie_asterisk_count'] = self.df_c['Cookie'].apply(lambda i: i.count('*'))
        # self.df_c['cookie_backtick_count'] = self.df_c['Cookie'].apply(lambda i: i.count('`'))
        # self.df_c['cookie_tilde_count'] = self.df_c['Cookie'].apply(lambda i: i.count('~'))
        # self.df_c['cookie_ampersand_count'] = self.df_c['Cookie'].apply(lambda i: i.count('&'))
        # self.df_c['cookie_exclamation_count'] = self.df_c['Cookie'].apply(lambda i: i.count('!'))
        # self.df_c['special_characters'] = self.df_c['Cookie'].apply(lambda i: self.ss_count(i))
        # del self.df_c['AcceptHdr']
        # self.df_c["Request_code"] = self.lb_make.fit_transform(self.df_c["Request"])
        # self.df_c["Request_code"].value_counts()
        # del self.df_c['Request']

        # self.df_c["Encoding_code"] = self.lb_make.fit_transform(self.df_c["Encoding"])
        # self.df_c["Encoding_code"].value_counts()
        # del self.df_c['Encoding']

        # self.df_c["Lang_code"] = self.lb_make.fit_transform(self.df_c["Lang"])
        # self.df_c["Lang_code"].value_counts()
        # del self.df_c['Lang']

        # self.df_c["Agent_code"] = self.lb_make.fit_transform(self.df_c["Agent"])
        # self.df_c["Agent_code"].value_counts()
        # del self.df_c['Agent']

        # del self.df_c['Cookie']
        # self.df_c["Cdata_code"] = self.lb_make.fit_transform(self.df_c["Cdata"])
        # self.df_c["Cdata_code"].value_counts()
        # del self.df_c['Cdata']

        # self.df_c['Clength'] = self.df_c['Clength'].astype('int64')
        # # Load the data
        # self.X_cookie = self.df_c[['Clength', 'Accept_Header_Length', 'Accept_Header_SubDirectory', 'Cookie_Length', 'cookie_less_than_count', 'cookie_open_brace_count', 'cookie_close_brace_count', 'cookie_plus_count', 'cookie_minus_count', 'cookie_double_quote_count',
        #                 'cookie_colon_count', 'cookie_semicolon_count', 'cookie_asterisk_count', 'cookie_backtick_count', 'cookie_tilde_count', 'cookie_ampersand_count', 'cookie_exclamation_count', 'special_characters', 'Request_code', 'Encoding_code', 'Lang_code', 'Agent_code', 'Cdata_code']]
        # self.dtest = xgb.DMatrix(self.X_cookie)

        # # Get predictions
        # self.y_pred_xgb_cookie = self.xgb_model.predict(self.dtest)
        # warnings.filterwarnings("ignore")
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # self.X_cookie = self.df_c[['Clength', 'Accept_Header_Length', 'Accept_Header_SubDirectory', 'Cookie_Length', 'cookie_less_than_count', 'cookie_open_brace_count', 'cookie_close_brace_count', 'cookie_plus_count', 'cookie_minus_count', 'cookie_double_quote_count',
        #                 'cookie_colon_count', 'cookie_semicolon_count', 'cookie_asterisk_count', 'cookie_backtick_count', 'cookie_tilde_count', 'cookie_ampersand_count', 'cookie_exclamation_count', 'special_characters', 'Request_code', 'Encoding_code', 'Lang_code', 'Agent_code', 'Cdata_code']]
        # self.X_cookie = np.reshape(self.X_cookie.values, (1, 23, 1))

        # # CNN PREDICTION

        # self.y_pred_cnn_cookie = self.cnn_model.predict(self.X_cookie)
        # self.y_pred_cnn_cookie = np.round(self.y_pred_cnn_cookie)

        # # LSTM PREDICION
        # self.X_cookie = np.reshape(self.X_cookie, (self.X_cookie.shape[0], 1, self.X_cookie.shape[1]))
        # self.y_pred_lstm_cookie = self.lstm_model.predict(self.X_cookie)
        # self.y_pred_lstm_cookie = np.round(self.y_pred_cnn_cookie)

        # URL prediction
        self.suspicious = 0

        # assuming your dataframe is called df and the URL column is called url
        try:
            self.df['URL'] = self.df['URL'].apply(lambda x: 'http://www.example.com/' + x.split('/', 3)[3])
        except:
            self.suspicious += 1

        try:
            self.df['use_of_ip'] = self.df['URL'].apply(lambda i: self.having_ip_address(i))
            del self.df['use_of_ip']
        except:
            self.suspicious += 1
            print("use of ip")

        # assuming df is defined somewhere earlier in your code
        self.df['abnormal_url'] = self.df['URL'].apply(lambda i: self.abnormal_url(i))
        del self.df['abnormal_url']

        self.df['period_count'] = self.df['URL'].apply(lambda i: i.count('.'))
        self.df['www_count'] = self.df['URL'].apply(lambda i: i.count('www'))
        self.df['at_count'] = self.df['URL'].apply(lambda i: i.count('@'))
        self.df['directory_count'] = self.df['URL'].apply(lambda i: self.no_of_dir(i))
        self.df['embedded_domain_count'] = self.df['URL'].apply(lambda i: self.no_of_embed(i))
        self.df['is_short_url'] = self.df['URL'].apply(lambda i: self.shortening_service(i))

        self.df['less_than_count'] = self.df['URL'].apply(lambda i: i.count('<'))
        self.df['open_brace_count'] = self.df['URL'].apply(lambda i: i.count('{'))
        self.df['close_brace_count'] = self.df['URL'].apply(lambda i: i.count('}'))
        self.df['plus_count'] = self.df['URL'].apply(lambda i: i.count('+'))
        self.df['minus_count'] = self.df['URL'].apply(lambda i: i.count('-'))
        self.df['double_quote_count'] = self.df['URL'].apply(lambda i: i.count('"'))
        self.df['colon_count'] = self.df['URL'].apply(lambda i: i.count(':'))
        self.df['semicolon_count'] = self.df['URL'].apply(lambda i: i.count(';'))
        self.df['asterisk_count'] = self.df['URL'].apply(lambda i: i.count('*'))
        self.df['backtick_count'] = self.df['URL'].apply(lambda i: i.count('`'))
        self.df['tilde_count'] = self.df['URL'].apply(lambda i: i.count('~'))
        self.df['ampersand_count'] = self.df['URL'].apply(lambda i: i.count('&'))
        self.df['exclamation_count'] = self.df['URL'].apply(lambda i: i.count('!'))
        self.df['digit_count'] = self.df['URL'].apply(lambda i: self.digit_count(i))
        self.df['special_char_count'] = self.df['URL'].apply(lambda i: self.ss_count(i))

        self.df['percent_count'] = self.df['URL'].apply(lambda i: i.count('%'))
        self.df['question_mark_count'] = self.df['URL'].apply(lambda i: i.count('?'))
        self.df['equal_sign_count'] = self.df['URL'].apply(lambda i: i.count('='))
        # Length of URL
        self.df['url_length'] = self.df['URL'].apply(lambda i: len(str(i)))
        # Hostname Length
        self.df['hostname_length'] = self.df['URL'].apply(lambda i: len(urlparse(i).netloc))
        self.df['iocs_count'] = self.df['URL'].apply(lambda i: self.iocs(i))

        del self.df['is_short_url']
        del self.df['hostname_length']
        self.X = self.df[['period_count', 'www_count', 'at_count', 'directory_count', 'embedded_domain_count',
                'less_than_count', 'open_brace_count', 'close_brace_count', 'plus_count',
                'minus_count', 'double_quote_count', 'colon_count', 'semicolon_count',
                'asterisk_count', 'backtick_count', 'tilde_count', 'ampersand_count',
                'exclamation_count', 'digit_count', 'special_char_count', 'percent_count',
                'question_mark_count', 'equal_sign_count', 'url_length', 'iocs_count']]
        self.y_pred_lgb = self.lgb_model_prediction(self.X)
        print("ok")
        self.y_pred_xgb = self.xgb_model_prediction(self.X)
        print("ok")
        self.y_pred_gbdt = self.gbdt_model_prediction(self.X)
        print("ok")
        self.y_pred_rf = self.rf_model_prediction(self.X)
        print("ok")
        # self.y_pred_xgb_cookie = 1 if self.y_pred_xgb_cookie == 0 else 0
        # assuming y_pred_lgb, y_pred_xgb, and y_pred_gbdt are arrays of 0s and 1s
        self.confidences = (self.y_pred_lgb + self.y_pred_xgb + self.y_pred_gbdt + self.y_pred_rf) / 4.0
        # self.confidences = (self.y_pred_lgb  + self.y_pred_gbdt + self.y_pred_rf + (self.y_pred_xgb_cookie*0.10)) / 4.0
        self.average_confidence = self.confidences.mean()
    def get_response(self):
        print(self.average_confidence)
        print (self.suspicious)
        if (self.average_confidence <=  0.5) and (self.suspicious == 0):
            return {"result":"Benign","Confidence":self.average_confidence,"Suspicious":self.suspicious}
        else:
            return {"result":"Malicious","Confidence":self.average_confidence,"Suspicious":self.suspicious}
    def ss_count(self,string):
        # Declaring variable for special characters
        special_char = 0

        for i in range(0, len(string)):
            ch = string[i]
            if (string[i].isalpha()):
                continue
            elif (string[i].isdigit()):
                continue
            else:
                special_char += 1
        return special_char
    def digit_count(self,url):
                digits = 0
                for i in url:
                    if i.isnumeric():
                        digits = digits + 1
                return digits

    def iocs(self,url):
        xss_and_sql_keywords = (
        # XSS attack keywords
        "<script>", "alert(", "onmouseover", "onload", "onclick", "onerror",
        "eval(", "document.cookie", "window.location", "innerHTML", "fromCharCode(",
        "encodeURIComponent(", "setTimeout(", "setInterval(", "xhr.open(", "xhr.send(",
        "parent.frames[", "prompt(", "confirm(", "formData.append(", "<img src=",
        "<audio src=", "<video src=", "<svg/onload=", "<marquee>", "<input type=\"text\" value=",
        "<a href=", "<link href=", "<iframe src=", "<body onload=", "<meta http-equiv=",
        "<form action=", "<textarea>", "<object data=", "<embed src=", "<style>", "<xss>", "<noscript>",
        "<applet>", "<base href=", "<s&#99;ript>", "al&#x65;rt(", "onmo&#x75;seover", "o&#x6e;load",
        "onclic&#x6b;", "onerror", "e&#x76;al(", "do&#x63;ument.cookie", "window.locat&#x69;on",
        "in&#x6e;erHTML", "fromCh&#x61;rCode(", "encodeURICompone&#x6e;t(", "setTim&#x65;out(",
        "setInt&#x65;rval(", "xhr.op&#x65;n(", "xhr.se&#x6e;d(", "parent.fr&#x61;mes[", "prom&#x70;t(",
        "confirm(", "formD&#x61;ta.append("
        # SQL injection keywords
        "OR", "AND", "--", ";", "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE",
        "EXECUTE", "UNION", "JOIN", "DROP", "CREATE", "ALTER", "TRUNCATE", "TABLE", "DATABASE",
        "HAVING", "LIKE", "ESCAPE", "ORDER BY", "GROUP BY", "LIMIT", "OFFSET", "XOR", "NOT",
        "BETWEEN", "IN", "EXISTS",
        # Encrypted keywords
        "<s&#99;ript>", "al&#x65;rt(", "onmo&#x75;seover", "o&#x6e;load", "onclic&#x6b;", "onerror",
        "e&#x76;al(", "do&#x63;ument.cookie", "window.locat&#x69;on", "in&#x6e;erHTML",
        "fromCh&#x61;rCode(", "encodeURICompone&#x6e;t(", "setTim&#x65;out(", "setInt&#x65;rval(",
        "xhr.op&#x65;n(", "xhr.se&#x6e;d(", "parent.fr&#x61;mes[", "prom&#x70;t(", "confirm(",
        "formD&#x61;ta.append(", "<img src=", "<audio src=",
            # XSS attack keywords
        "<script>","<script", "alert(", "onmouseover", "onload", "onclick", "onerror",
        "eval(", "document.cookie", "window.location", "innerHTML", "fromCharCode(",
        "encodeURIComponent(", "setTimeout(", "setInterval(", "xhr.open(", "xhr.send(",
        "parent.frames[", "prompt(", "confirm(", "formData.append(", "<img src=",
        "<audio src=", "<video src=", "<svg/onload=", "<marquee>", "<input type=\"text\" value=",
        "<a href=", "<link href=", "<iframe src=", "<body onload=", "<meta http-equiv=",
        "<form action=", "<textarea>", "<object data=", "<embed src=", "<style>", "<xss>", "<noscript>",
        "<applet>", "<base href=", "<s&#99;ript>", "al&#x65;rt(", "onmo&#x75;seover", "o&#x6e;load",
        "onclic&#x6b;", "onerror", "e&#x76;al(", "do&#x63;ument.cookie", "window.locat&#x69;on",
        "in&#x6e;erHTML", "fromCh&#x61;rCode(", "encodeURICompone&#x6e;t(", "setTim&#x65;out(",
        "setInt&#x65;rval(", "xhr.op&#x","xhr.send(", "parent.fra&#x6d;es[", "pro&#x6d;pt(", "con&#x66;irm(", "formD&#x61;ta.append(",
        "<img s&#x72;c=", "<audio s&#x72;c=", "<video s&#x72;c=", "<svg/onload=", "<ma&#x72;quee>",
        "<inpu&#x74; type=\"text\" value=", "<a hre&#x66;=", "<link hre&#x66;=", "<iframe s&#x72;c=",
        "<body onl&#x6f;ad=", "<meta http-equiv=", "<form action=", "<texta&#x72;ea>", "<ob&#x6a;ect data=",
        "<embed s&#x72;c=", "<style>", "<xss>", "<noscript>", "<applet>", "<base href=", "<s&#99;ript>",
        # SQL injection keywords
        "OR", "AND", "--", ";", "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "EXECUTE", "UNION",
        "JOIN", "DROP", "CREATE", "ALTER", "TRUNCATE", "TABLE", "DATABASE", "HAVING", "LIKE", "ESCAPE",
        "ORDER BY", "GROUP BY", "LIMIT", "OFFSET", "XOR", "NOT", "BETWEEN", "IN", "EXISTS", "OR 1=1", "AND 1=1",
        "--", ";", "'", "\"", "`", "/**/", "/*!*/", "/*...*/", "|", "^",
        "; SELECT * FROM users WHERE username='admin' --", "1'; DROP TABLE users; --",
        "UNION SELECT 1,2,3,4,5,6,7,8,9,10 FROM users WHERE username='admin'",
        "SELECT * FROM users WHERE id = 1 OR 1=1", "SELECT * FROM users WHERE username='admin' AND password='password'",
        "SELECT * FROM users WHERE username LIKE '%admin%'", "SELECT * FROM users WHERE username IN ('admin', 'user', 'guest')",
        "SELECT * FROM users WHERE EXISTS (SELECT * FROM admin_users WHERE username='admin')",
        "SELECT * FROM users WHERE password=MD5('password')", "SELECT * FROM users WHERE password=SHA1('password')",
        "SELECT * FROM users WHERE password=SHA2('password', 256)", "SELECT * FROM users WHERE password=PASSWORD('password')",
        # XSS keywords
        "<sc&#x72;ipt>", "<img onerror=", "<svg/onload=alert(", "<audio onloadedmetadata=",
        "<video onloadedmetadata=", "<iframe srcdoc=", "<form onsubmit=alert(", "<object type=text/html data=",
        "<applet codebase=", "<link rel=", "<base href=", "<meta charset=", "<textarea onfocus=",
        "<body onload=", "<input type=\"text\" value=\"", "<a href=", "<embed src=", "<style>",
        "<script>alert('xss')</script>", "<img src=x onerror=alert('xss')>", "<body onload=alert('xss')>",
        "<a href=\"javascript:alert('xss')\">Click Here</a>", "<iframe src=\"javascript:alert('xss')\"></iframe>",
        "<script>alert(String.fromCharCode(88,83,83))</script>", "<input value=\"\" onclick=alert('xss')>",
        # SQL injection keywords
        "AND 1=1", "OR 1=1", "AND 1=2", "OR 1=2", "SELECT COUNT(*) FROM", "SELECT * FROM users WHERE",
        "SELECT * FROM users ORDER BY", "SELECT * FROM users LIMIT", "SELECT * FROM users OFFSET",
        "SELECT * FROM users WHERE 1=1", "SELECT * FROM users WHERE 1=0", "SELECT * FROM users WHERE id=",
        "SELECT * FROM users WHERE username=", "SELECT * FROM users WHERE password=",
        "SELECT * FROM users WHERE email=", "SELECT * FROM users WHERE status=",
        "SELECT * FROM users WHERE role=", "SELECT * FROM users WHERE access_token=",
        "SELECT * FROM users WHERE refresh_token=", "SELECT * FROM users WHERE session_id=",
        "INSERT INTO users (id, username, password, email, status, role, access_token, refresh_token, session_id) VALUES",
        "UPDATE users SET", "DELETE FROM users WHERE id=", "DROP TABLE", "DROP DATABASE",
        "CREATE DATABASE", "CREATE TABLE", "ALTER TABLE", "TRUNCATE TABLE", "UNION SELECT",
        "HAVING 1=1", "HAVING 1=0", "LIKE '%", "LIKE '%admin%'",

        # #cmd via usman
        # 'tracert','ping','nmap','ncat','nikto','ssh','trace'
    )

        p = (sum(url.count(x) for x in xss_and_sql_keywords))
        self.suspicious += p
        print("keyword used")
        return (p)
    def lgb_model_prediction(self,X):
        y_pred_lgb = self.lgb_model.predict(X)
        return y_pred_lgb

    # Load XGBoost model
    def xgb_model_prediction(self,X):
        y_pred_xgb = self.xgb_model_2.predict(X)
        return y_pred_xgb
    # Load Gradient Boosting model

    def gbdt_model_prediction(self,X):
        y_pred_gbdt = self.gbdt_model.predict(X)
        return y_pred_gbdt

    # Load random forest model
    def rf_model_prediction(self,X):
        y_pred_rf = self.rf_model.predict(X)
        return y_pred_rf
    def process_input_dict(self):
        # Create dataframe from input dictionary
        df = pd.DataFrame.from_dict([self.dict])

        # Create dataframe with remaining columns
        df_c = df.drop('URL', axis=1)

        # Return dataframe
        return df[['URL']], df_c
    def no_of_dir(self,url):
        urldir = urlparse(url).path
        return urldir.count('/')
    def having_ip_address(self,url):
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
            # IPv4 in hexadecimal
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
        if match:
            return 1
        else:
            return 0
    def no_of_embed(self,url):
            urldir = urlparse(url).path
            return urldir.count('//')
    def abnormal_url(self,url):
            try:
                hostname = urlparse(url).hostname
                hostname = str(hostname)
                match = re.search(hostname, url)
                if match:
                    return 1
                else:
                    return 0
            except Exception as e:
                self.suspicious += 1
                print("not a host name")
                return -1  # return -1 to indicate an error occurred
    def shortening_service(self,url):
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                        'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                        'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                        'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                        'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                        'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                        'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                        'tr\.im|link\.zip\.net',
                        url)
        if match:
            return 1
        else:
            return 0
from flask import Flask, render_template, request, jsonify

predictor = Predict()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_url():
    url = request.json['URL']  # Get the URL from the POST request
    predictor.pred({'URL': url})  # Make a prediction using your class
    response = predictor.get_response()  # Get the response from your prediction class
    print (response)
    return jsonify({'prediction': response['result'] , 'confidence': response['Confidence'] , 'suspicious': response['Suspicious']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=False)
