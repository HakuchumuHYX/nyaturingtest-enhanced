# nyaturingtest-enhanced

## 项目简介

本项目基于 [nonebot-plugin-nyaturingtest](https://github.com/shadow3aaa/nonebot-plugin-nyaturingtest) 进行修改。

## 配置指南

如果你想使用本插件，建议先从[原项目](https://github.com/shadow3aaa/nonebot-plugin-nyaturingtest)安装所需的依赖。  
如果你是Windows系统，可能会出现运行报错，可以带着你的错误日志在[这里](https://gemini.google.com)或是与之类似的地方求助。  
本插件的配置参数与原项目一致，详细说明请查阅[原项目文档](https://github.com/shadow3aaa/nonebot-plugin-nyaturingtest/blob/master/README.md)。

## 功能变更

与原项目相比，本项目主要做出了以下更改：

1. 增加了对回复消息的处理，以及可以回复他人的指定信息。
2. 修改了发言欲望，使bot不会频率过高发送信息。
3. 优化了处理逻辑，尽量使事件异步进行，减少可能出现的阻塞情况。
4. 引入sqlite代替json，更加优化性能。

## 注意事项

~~由于该插件的运作逻辑，导致token消耗会**非常非常大**（在一个一天发2k条消息的群聊中使用，一天总共约消耗30Mtoken），使用请注意。~~  
~~也许在之后会优化一下token使用。~~  
更改了feedback逻辑，以轻微降低智能的代价大幅降低了token使用。  
更改了长期记忆索引的逻辑，大幅降低了token使用。  
目前的token用量大约在先前的20%左右，即处理大概2k条消息总共消耗6-7Mtoken。  
建议：.env文件中配置的对话用大模型用优质一点的（作为参考，我用的是DeepSeek-V3.2-Exp），会更加智能。  

## Special Thanks
[G指导](https://gemini.google.com)
