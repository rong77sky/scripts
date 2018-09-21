# coding: utf-8

import logging
from binascii import b2a_hex, a2b_hex
from Crypto.Cipher import AES

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y-%b-%d %H:%M:%S',
                    filename='log.log',)

class MyCrypt():
    def __init__(self, key):
        self.key = key
        self.mode = AES.MODE_CBC

    def encrypt(self, text):
        cryptor = AES.new(self.key, self.mode, self.key)
        #这里密钥key 长度必须为16（AES-128）、24（AES-192）、或32（AES-256）Bytes 长度.目前AES-128足够用
        length = 16
        count = len(text)
        add = length - (count % length)
        text = text + ('\0' * add)
        self.ciphertext = cryptor.encrypt(text)
        #因为AES加密时候得到的字符串不一定是ascii字符集的，输出到终端或者保存时候可能存在问题
        #所以这里统一把加密后的字符串转化为16进制字符串
        return b2a_hex(self.ciphertext)

    def decrypt(self, text):
        cryptor = AES.new(self.key, self.mode, self.key)
        plain_text = cryptor.decrypt(a2b_hex(text))
        return plain_text.rstrip('\0')

if __name__ == '__main__':
    # msg = 'http://10.168.100.198:8983/solr/'
    # crypt_ins = MyCrypt('qacorpautohomeco')
    msg = '10.27.4.81'
    crypt_ins = MyCrypt('qacorpautohomeco')
    en_text = crypt_ins.encrypt(msg)
    print en_text
    de_text = crypt_ins.decrypt(en_text)
    print de_text
