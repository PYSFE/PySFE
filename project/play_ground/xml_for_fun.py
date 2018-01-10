import os
import io
import xmltodict

if __name__ == '__main__':
    xml_file = r'C:\Users\Ian Fu\Dropbox (OFR-UK)\Bicester_team_projects\Live_projects\Trienus\B-RISK trail 2\riskdata\basemodel_(vent_15)_(growth_270)'
    xml_file = os.path.join(xml_file, 'output1.xml')

    with io.open(xml_file, 'r', encoding='utf-8') as f:
        xml_string = f.read()

    dict_ = xmltodict.parse(xml_string)

    res = dict_['output']['run']['room']['0']['time']

    for k, v in dict_.items():
       print('key:', k, '; val:', v)

