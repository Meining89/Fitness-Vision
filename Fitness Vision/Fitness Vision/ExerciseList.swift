//
//  ExerciseList.swift
//  Fitness Vision
//
//  Created by Emma Fu on 2023-09-18.
//

import Foundation
import SwiftUI

struct ExercisesPage: View {
    var body: some View {
        NavigationView {
            ZStack{
                LinearGradient(gradient: Gradient(colors: [.blue, .white]), startPoint: .topLeading, endPoint: .bottomTrailing)
                    .edgesIgnoringSafeArea(.all)
                VStack{
                    Text("**Exercises**")
                        .foregroundStyle(.white)
                        .font(.largeTitle)
                        .multilineTextAlignment(.center)
                        .padding()
                    Spacer()
                }
            }
        }
    }
}

struct ExercisesPage_Previews: PreviewProvider {
    static var previews: some View {
        ExercisesPage()
    }
}
