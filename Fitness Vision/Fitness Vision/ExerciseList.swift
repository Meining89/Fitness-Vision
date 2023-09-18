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
            VStack{
                Text("Exercises")
                    .font(.largeTitle)
                    .multilineTextAlignment(.center)
                    .padding()
                Spacer()
            }
        }
    }
}

struct ExercisesPage_Previews: PreviewProvider {
    static var previews: some View {
        ExercisesPage()
    }
}
