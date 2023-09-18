//
//  ExerciseList.swift
//  Fitness Vision
//
//  Created by Emma Fu on 2023-09-18.
//

import Foundation
import SwiftUI

struct ExerciseInfo: Identifiable{
    let id = UUID()
    let name: String
    let description: String
}

struct ExerciseRow: View {
    @State private var isExpanded = false
    let exercise: ExerciseInfo
    
    var body: some View {
        VStack(alignment: .leading) {
            Button(action: {
                isExpanded.toggle()
            }) {
                Text(exercise.name)
                    .font(.headline)
            }
            
            if isExpanded {
                Text(exercise.description)
                    .font(.body)
                    .foregroundColor(.secondary)
            }
        }
    }
}

struct ExercisesPage: View {
    
    init() {
            // Use this if NavigationBarTitle is with Large Font
            UINavigationBar.appearance().largeTitleTextAttributes = [.foregroundColor: UIColor.white]

            // Use this if NavigationBarTitle is with displayMode = .inline
            UINavigationBar.appearance().titleTextAttributes = [.foregroundColor: UIColor.white]
    }
    
    var body: some View {
        // Hardcoded exercise list
        let exerciseList: [ExerciseInfo] = [
            ExerciseInfo(name: "Push Up", description: "Push up description"),
            ExerciseInfo(name: "Squat", description: "Squat description")
        ]
        
        
        NavigationView {
            ZStack{
                LinearGradient(gradient: Gradient(colors: [.blue, .white]), startPoint: .topLeading, endPoint: .bottomTrailing)
                    .edgesIgnoringSafeArea(.all)
                
                VStack{
                    List(exerciseList) { exercise in
                        ExerciseRow(exercise: exercise)
                    }
                    .scrollContentBackground(.hidden)
                    .navigationTitle("Available Exercises")
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
