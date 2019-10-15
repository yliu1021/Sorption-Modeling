//
//  ViewController.swift
//  GuiLearning
//
//  Created by Yuhan Liu on 7/16/19.
//  Copyright © 2019 Yuhan Liu. All rights reserved.
//

import Cocoa
import Charts

class ViewController: NSViewController {

    let whiteImage = NSImage(named: "white")
    let blackImage = NSImage(named: "black")
    var buttons = [NSButton]()

    var fileTextField: NSTextField!
    
    var lineChart: LineChartView!
    
    @objc func chooseFile(sender: NSButton) {
        let dialog = NSOpenPanel();
        
        dialog.title                   = "Choose a grid file";
        dialog.showsResizeIndicator    = true;
        dialog.showsHiddenFiles        = false;
        dialog.canChooseDirectories    = false;
        dialog.canCreateDirectories    = false;
        dialog.allowsMultipleSelection = false;
        dialog.allowedFileTypes        = ["csv"];
        
        if (dialog.runModal() == NSApplication.ModalResponse.OK) {
            let result = dialog.url // Pathname of the file
            
            if (result != nil) {
                let path = result!.path
                loadFromFile(gridFile: path)
                startDFT(sender: nil)
            }
        } else {
            // User clicked on "Cancel"
            return
        }
    }
    
    func loadFromFile(gridFile: String) {
        var csvLocation = gridFile
        
        let fileManager = FileManager.default
        let baseDir = "/Users/yuhanliu/Google Drive/1st year/Research/sorption_modeling/"
        
        if !fileManager.fileExists(atPath: gridFile) {
            csvLocation = baseDir + gridFile
            if !fileManager.fileExists(atPath: csvLocation) {
                print("File doesn't exist")
                return
            } else {
                
            }
        }
        
        guard let string = try? String(contentsOfFile: csvLocation) else {
            print("Could not open file")
            return
        }

        var index = 0
        
        for char in string {
            if char == "1" {
                buttons[index].image = blackImage
                index += 1
            } else if char == "0" {
                buttons[index].image = whiteImage
                index += 1
            }
        }
    }
    
    func getValueFrom(button: NSButton) -> Double {
        guard let name = button.image?.name() else {
            return 0.0
        }
        
        if name == "black" {
            return 1.0
        } else {
            return 0.0
        }
    }
    
    func getMetricFrom(curve: [Double]) -> Double {
        var metric = 0.0
        for (i, d) in curve.enumerated() {
            metric += abs(d - Double(i) * STEP_SIZE)
        }
        return metric
    }
    
    func show(_ density: [Double]) {
        let adsorption = density[..<N_ADSORP]
//        let desorption = density[N_ADSORP...]
        
        let baseLine = [ChartDataEntry(x: 0, y: 0), ChartDataEntry(x: 1, y: 1)]
        let adsorptionDataEntry = adsorption.enumerated().map { x, y in
            return ChartDataEntry(x: Double(x) * STEP_SIZE, y: y)
        }
//        let desorptionDataEntry = desorption.enumerated().map { x, y in
//            return ChartDataEntry(x: Double(x) * STEP_SIZE, y: y)
//        }
        
        let data = LineChartData()
        let baseLineDataSet = LineChartDataSet(entries: baseLine, label: "Baseline")
        baseLineDataSet.colors = [NSUIColor.black]
        baseLineDataSet.drawCirclesEnabled = false
        baseLineDataSet.valueTextColor = .clear
        data.addDataSet(baseLineDataSet)

        let adsorptionDataSet = LineChartDataSet(entries: adsorptionDataEntry, label: "Adsorption")
        adsorptionDataSet.colors = [NSUIColor.blue]
        adsorptionDataSet.drawCirclesEnabled = false
        adsorptionDataSet.valueTextColor = .clear
        data.addDataSet(adsorptionDataSet)
//        let desorptionDataSet = LineChartDataSet(entries: desorptionDataEntry, label: "Desorption")
//        desorptionDataSet.colors = [NSUIColor.green]
//        data.addDataSet(desorptionDataSet)
        
        let metric = getMetricFrom(curve: Array(adsorption)) / 20.0
        lineChart.data = data
        lineChart.chartDescription?.text = String(format: "Metric: %.4f", metric)
    }
    
    @objc func startDFT(sender: NSButton?) {
        var grid = [Double]()
        grid.reserveCapacity(GRID_SIZE)
        for button in buttons {
            let buttonVal = getValueFrom(button: button)
            grid.append(buttonVal)
        }
        
        let density = runDFT(grid: grid)
        show(density)
    }
    
    @objc func gridCellPressed(sender: NSButton) {
        let buttonVal = getValueFrom(button: sender)
        
        if buttonVal > 0.5 {
            sender.image = whiteImage
        } else {
            sender.image = blackImage
        }
        
        startDFT(sender: nil)
    }
    
    @objc func loadFile(sender: NSButton) {
        let filePath = fileTextField.stringValue
        loadFromFile(gridFile: filePath)
        startDFT(sender: nil)
    }
    
    override func viewWillAppear() {
        self.view.window?.isMovableByWindowBackground = true
        startDFT(sender: nil)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupNL()
        let visualEffect = NSVisualEffectView()
        visualEffect.translatesAutoresizingMaskIntoConstraints = false
        visualEffect.material = .mediumLight
        visualEffect.state = .active
        view.addSubview(visualEffect)
        
        visualEffect.leadingAnchor.constraint(equalTo: view.leadingAnchor).isActive = true
        visualEffect.trailingAnchor.constraint(equalTo: view.trailingAnchor).isActive = true
        visualEffect.topAnchor.constraint(equalTo: view.topAnchor).isActive = true
        visualEffect.bottomAnchor.constraint(equalTo: view.bottomAnchor).isActive = true

        buttons.reserveCapacity(N_SQUARES)
        let buttonSize = 20.0
        for y in 0..<GRID_SIZE {
            for x in 0..<GRID_SIZE {
                let frame = NSRect(x: x*buttonSize, y: y*buttonSize, width: buttonSize, height: buttonSize)
                let defaultImage = NSImage(named: "black")
                let button = NSButton(frame: frame)
                button.isBordered = false
                button.imagePosition = .imageOnly
                button.imageScaling = .scaleProportionallyUpOrDown
                button.image = defaultImage
                button.tag = x*20 + y
                button.target = self
                button.action = #selector(gridCellPressed)
                
                buttons.append(button)
                view.addSubview(button)
            }
        }
        let chooseFileButton = NSButton(title: "Choose grid", target: self, action: #selector(chooseFile(sender:)))
        chooseFileButton.frame = NSRect(x: 10, y: GRID_SIZE*buttonSize + 10, width: 100, height: 30)
        view.addSubview(chooseFileButton)
        
        lineChart = LineChartView(frame: CGRect(x: GRID_SIZE*buttonSize, y: 0, width: 400, height: GRID_SIZE*buttonSize))
        lineChart.backgroundColor = .white
        lineChart.gridBackgroundColor = .white
        view.addSubview(lineChart)
        
        view.frame = NSRect(x: 0, y: 0, width: GRID_SIZE*buttonSize + 400, height: GRID_SIZE*buttonSize + 60)
    }

}

