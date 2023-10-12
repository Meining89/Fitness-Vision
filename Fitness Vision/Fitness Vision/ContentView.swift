import SwiftUI

struct ContentView: View {
    var body: some View {
        // Version 1
        HostedViewController()
            .ignoresSafeArea()
        
        // Version 2
//        HostedCameraViewController()
//            .ignoresSafeArea()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
