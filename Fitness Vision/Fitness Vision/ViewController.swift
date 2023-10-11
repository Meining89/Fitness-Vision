//
//  CameraViewController.swift
//  Fitness Vision
//
//  Created by Emma Fu on 2023-10-11.
//
import Foundation
import SwiftUI
import CoreData
import AVFoundation
import UIKit

// https://github.com/barbulescualex/iOSCustomCamera/blob/master/Starter/CustomCamera/ViewController%2BExtras.swift
// https://www.youtube.com/watch?v=ZYPNXLABf3c&ab_channel=iOSAcademy


class CameraViewController: UIViewController {
    var captureSession : AVCaptureSession?
    
    var backCamera : AVCaptureDevice?
    var frontCamera : AVCaptureDevice?
    var backInput : AVCaptureInput?
    var frontInput : AVCaptureInput?
    
    var previewLayer : AVCaptureVideoPreviewLayer?
    var videoOutput : AVCaptureVideoDataOutput?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Request camera permission
        checkCameraPermission()
        
        // Set up and start the camera capture session
        setupCamera()
    }
    
    private func checkCameraPermission(){
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            // Camera access already granted
            return
        case .notDetermined:
            // Request camera permission
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted {
                    // Permission granted, setup camera
                    self?.setupCamera()
                } else {
                    // Permission denied, handle appropriately
                    return
                }
            }
        case .denied, .restricted:
            // Camera access denied or restricted, handle appropriately
            return
        @unknown default:
            return
        }
    }
    
    private func setupCamera(){
        guard let captureSession = captureSession else { return }
        
        // Define the camera input
        
        // Currently only front camera
        // For all camera, use line below
        // if let device = AVCaptureDevice.default(for: .video)
        if let frontCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) {
            do {
                let cameraInput = try AVCaptureDeviceInput(device: frontCamera)
                
                if captureSession.canAddInput(cameraInput) {
                    captureSession.addInput(cameraInput)
                    
                    // Define the video preview layer
                    previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
                    if let previewLayer = previewLayer {
                        view.layer.addSublayer(previewLayer)
                        previewLayer.videoGravity = .resizeAspectFill
                        previewLayer.frame = view.bounds
                    }
                    
                    // Start the capture session
                    captureSession.startRunning()
                    self.captureSession = captureSession
                }
            } catch {
                // Handle any errors
                print(error)
            }
        }
        
    }
    
    override func viewDidLayoutSubviews() {
         super.viewDidLayoutSubviews()
         previewLayer?.frame = view.bounds
     }
}

struct HostedViewController: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> UIViewController {
        return CameraViewController()
    }
    
    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        
    }
}
