# WSL2 Hand-out

## WSL 常用命令

- https://learn.microsoft.com/zh-cn/windows/wsl/basic-commands

## 如何在一台全新的WIN11上安装WSL2



1. 以管理员模式打开powershell

   ![image-20230604203430918](https://s2.loli.net/2023/06/04/PgOYt5bs2VzWREL.png)

2. 输入下面命令

   - 安装WSL

   ```powershell
   wsl --install
   ```

   - 配置

   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

   - 选择 WSL 2

   ```powershell
   wsl --set-default-version 2
   ```

   - **重启**

## 在WSL2 中安装Ubuntu20.04 到指定目录下

- 假设安装在文件夹 **C:\Linux** 下

- 进入文件夹 **C:\Linux** ,下载Ubuntu20.04（稍微等一会儿）

  ```powershell
  Invoke-WebRequest -Uri https://wsldownload.azureedge.net/Ubuntu_2004.2020.424.0_x64.appx -OutFile Ubuntu20.04.appx -UseBasicParsing 
  ```

- 接下来启动Ubuntu

  ```powershell
  Rename-Item .\Ubuntu20.04.appx Ubuntu.zip
  Expand-Archive .\Ubuntu.zip -Verbose
  cd .\Ubuntu\
  .\ubuntu2004.exe
  ```

## 如何将镜像文件移动到指定地方

## 导入/导出发行版

- 导出

```powershell
wsl --export <Distribution Name> <FileName>
```

- 导入

```powershell
wsl --import <Distribution Name> <InstallLocation> <FileName>
```

- **Distribution Name** 显示在wsl中的镜像名字，其实可以任意取
- **InstallLocation** 你想要让这个镜像放置的位置
- **FileName** 一个 .tar 文件

## 将WSL关机/重启

## 卸载发行版

尽管可以通过 Microsoft Store 安装 Linux 发行版，但无法通过 Store 将其卸载。

注销并卸载 WSL 发行版：

```powershell
wsl --unregister <DistributionName>
```





