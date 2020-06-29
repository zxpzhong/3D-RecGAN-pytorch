// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;

// public class rotation : MonoBehaviour
// {
//     // Update is called once per frame
//     int speed = 50;
//     Vector3 target = new Vector3(0.5f, 0.5f, 0.5f);

//     // Start is called before the first frame update
//     void Start()
//     {
        
//     }
//     void Update () {
//         // transform.Rotate(Vector3.forward, 10.0f * Time.deltaTime);
//         transform.RotateAround(target, Vector3.up,  speed*Time.deltaTime);
//     }
// }


using UnityEngine;
using System;
using System.IO;
public class rotation : MonoBehaviour
{
    int speed = 50;
    Vector3 target = new Vector3(0.5f, 0.5f, 0.5f);
    public Camera cam;
    public RenderTexture rt;
    public float angle=0;
    public void Start()
    {
        if(cam==null)
        {
            cam = this.GetComponent<Camera>();
        }
    }
    private void Update()
    {
        if(cam==null)
        { return; }
        if(Input.GetKeyDown(KeyCode.F4))
        {
            _SaveCamTexture();
        }
        transform.RotateAround(target, Vector3.up,  speed*Time.deltaTime);
        angle+=speed*Time.deltaTime;
        
        if(angle <= 360)
        {
            Debug.Log("当前转角"+angle);
            _SaveCamTexture();
        }
        
    }
    private void _SaveCamTexture()
    {
        rt = cam.targetTexture;
        if(rt!=null)
        {
            _SaveRenderTexture(rt);
            rt = null;
        }
        else
        {
            GameObject camGo = new GameObject("camGO");
            Camera tmpCam = camGo.AddComponent<Camera>();
            tmpCam.CopyFrom(cam);
           // rt = new RenderTexture(Screen.width, Screen.height, 16, RenderTextureFormat.ARGB32);
            rt =RenderTexture.GetTemporary(Screen.width,Screen.height,16, RenderTextureFormat.ARGB32);

            tmpCam.targetTexture = rt;
            tmpCam.Render();
            _SaveRenderTexture(rt);
            Destroy(camGo);
            //rt.Release();
            RenderTexture.ReleaseTemporary(rt);
            //Destroy(rt);
            rt = null;
        }

    }
    private void _SaveRenderTexture(RenderTexture rt)
    {
        RenderTexture active = RenderTexture.active;
        RenderTexture.active = rt;
        Texture2D png = new Texture2D(rt.width, rt.height, TextureFormat.ARGB32, false);
        png.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        png.Apply();
        RenderTexture.active = active;
        byte[] bytes = png.EncodeToJPG();
        string path = string.Format("Assets/../rt_{0}_{1}_{2}.jpg", DateTime.Now.Hour, DateTime.Now.Minute, DateTime.Now.Second);
        FileStream fs = File.Open(path, FileMode.Create);
        BinaryWriter writer = new BinaryWriter(fs);
        writer.Write(bytes);
        writer.Flush();
        writer.Close();
        fs.Close();
        Destroy(png);
        png = null;
        Debug.Log("保存成功！"+path);
    }
}
